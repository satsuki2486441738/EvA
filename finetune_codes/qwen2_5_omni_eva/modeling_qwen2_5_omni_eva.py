# coding=utf-8
"""Qwen2.5-Omni-EvA model.

This keeps the official Qwen2.5-Omni Thinker text/audio path intact and adds
CED as a residual audio stream at the audio placeholder embeddings.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerTextModel,
)

from eva_processor import CEDProcessor, resample_proc_to_whisper_timeaware

from .configuration_qwen2_5_omni_eva import Qwen2_5OmniEvaConfig


logger = logging.get_logger(__name__)


class Qwen2_5OmniEvaForConditionalGeneration(
    Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin
):
    config_class = Qwen2_5OmniEvaConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _no_split_modules = ["Qwen2_5OmniAudioEncoder"]
    _keys_to_ignore_on_save = [r"ced_model\..*"]
    _keys_to_ignore_on_load_unexpected = [r"ced_model\..*"]

    def __init__(self, config: Qwen2_5OmniEvaConfig):
        super().__init__(config)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(config.audio_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None

        self.ced_processor = CEDProcessor(config) if config.use_ced_feature else None
        self.ced_model = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_ced_model(self, ced_model):
        self.ced_model = ced_model
        if ced_model is None:
            return
        try:
            dev = next(self.model.parameters()).device
        except Exception:
            dev = None
        if dev is not None:
            self.ced_model.to(device=dev, dtype=torch.float32).eval()
        else:
            self.ced_model.to(dtype=torch.float32).eval()
        for param in self.ced_model.parameters():
            param.requires_grad = False

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Qwen2_5OmniThinkerTextModel):
            module.gradient_checkpointing = value
        if isinstance(module, Qwen2_5OmniAudioEncoder):
            module.gradient_checkpointing = value

    def _initialize_newly_added_modules(self):
        if self.ced_processor is None:
            return
        nn.init.constant_(self.ced_processor.alpha, 0.1)
        for m in self.ced_processor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        gate = getattr(getattr(self.ced_processor, "audio_aggregator", None), "gate", None)
        if gate is not None:
            last = list(gate.children())[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        return_lengths: bool = False,
        **kwargs,
    ):
        valid_rows = None
        try:
            tower_device = next(self.audio_tower.parameters()).device
            input_features = input_features.to(tower_device)
            if feature_attention_mask is not None:
                feature_attention_mask = feature_attention_mask.to(tower_device)
            if audio_feature_lengths is not None:
                audio_feature_lengths = audio_feature_lengths.to(tower_device)
        except Exception:
            pass

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            valid = audio_feature_lengths > 0
            if not valid.any():
                if return_lengths:
                    return None, None, None, None
                return None
            valid_rows = valid.nonzero(as_tuple=True)[0]
            if not valid.all():
                logger.warning(f"Dropping {(~valid).sum().item()} sample(s) with zero audio feature length")
                input_features = input_features[valid]
                feature_attention_mask = feature_attention_mask[valid]
                audio_feature_lengths = audio_feature_lengths[valid]
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            if audio_feature_lengths is None:
                raise ValueError("audio_feature_lengths is required when feature_attention_mask is None")
            valid_rows = torch.arange(audio_feature_lengths.shape[0], device=audio_feature_lengths.device)

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
            return_dict=True,
            **kwargs,
        )
        if audio_outputs.last_hidden_state.shape[0] != int(audio_output_lengths.sum().item()):
            raise ValueError("length of audio_features should match audio_output_lengths")
        if return_lengths:
            return (
                audio_outputs.last_hidden_state,
                audio_output_lengths,
                audio_feature_lengths,
                valid_rows.to(audio_outputs.last_hidden_state.device),
            )
        return audio_outputs

    def _scatter_audio_features_by_sample(self, inputs_embeds, audio_features, audio_lengths, audio_mask, valid_rows):
        rows_with_tokens = audio_mask.any(dim=1).nonzero(as_tuple=True)[0].to(valid_rows.device)
        missing_rows = rows_with_tokens[~torch.isin(rows_with_tokens, valid_rows)]
        if missing_rows.numel() > 0:
            raise ValueError(
                "Audio tokens exist for samples with no valid audio features: "
                f"{missing_rows.detach().cpu().tolist()}"
            )

        offset = 0
        hidden_size = inputs_embeds.shape[-1]
        for feat_idx, batch_idx in enumerate(valid_rows.tolist()):
            length = int(audio_lengths[feat_idx].item())
            pos = audio_mask[batch_idx].nonzero(as_tuple=True)[0]
            if len(pos) != length:
                raise ValueError(
                    f"Audio token/features mismatch for sample {batch_idx}: "
                    f"tokens={len(pos)}, features={length}"
                )
            if length == 0:
                continue
            chunk = audio_features[offset : offset + length]
            if chunk.numel() != length * hidden_size:
                raise ValueError(
                    f"Invalid audio feature slice for sample {batch_idx}: "
                    f"expected {length * hidden_size} values, got {chunk.numel()}"
                )
            inputs_embeds[batch_idx, pos] = chunk
            offset += length
        if offset != audio_features.shape[0]:
            raise ValueError(f"Unused audio features: consumed={offset}, total={audio_features.shape[0]}")

    def _add_ced_features(
        self,
        inputs_embeds,
        input_ids,
        token_audio_mask,
        audio_waveform,
        audio_waveform_lengths,
        audio_feature_lengths,
        audio_output_lengths,
        valid_rows,
    ):
        if (
            audio_waveform is None
            or audio_waveform.numel() == 0
            or self.ced_model is None
            or self.ced_processor is None
            or token_audio_mask is None
        ):
            return inputs_embeds

        ced_features, ced_rows = self._extract_ced_features(audio_waveform, audio_waveform_lengths)
        if ced_features is None:
            return inputs_embeds

        output_by_row = {
            int(batch_idx): int(audio_output_lengths[i].item())
            for i, batch_idx in enumerate(valid_rows.detach().cpu().tolist())
        }
        mel_by_row = {
            int(batch_idx): int(audio_feature_lengths[i].item())
            for i, batch_idx in enumerate(valid_rows.detach().cpu().tolist())
        }
        for feat_idx, batch_idx in enumerate(ced_rows.detach().cpu().tolist()):
            pos = token_audio_mask[batch_idx].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            feat_len = output_by_row.get(batch_idx, len(pos))
            t_mel = mel_by_row.get(batch_idx, feat_len * 4)
            if feat_len != len(pos):
                logger.warning(
                    f"CED/audio token length mismatch for sample {batch_idx}: "
                    f"tokens={len(pos)}, audio_output={feat_len}; using token count"
                )
                feat_len = len(pos)
            resampled = resample_proc_to_whisper_timeaware(
                x_t=ced_features[feat_idx],
                feat_len=feat_len,
                T_mel=t_mel,
                step_mel=max(1, t_mel // max(1, feat_len)),
                center_mel=max(0, (t_mel // max(1, feat_len)) // 2),
            )
            inputs_embeds[batch_idx, pos] += resampled.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
        return inputs_embeds

    def _extract_ced_features(self, audio_waveform, audio_waveform_lengths=None):
        try:
            try:
                ced_device = next(self.ced_model.parameters()).device
            except Exception:
                ced_device = audio_waveform.device
            if audio_waveform_lengths is None:
                lengths = torch.full(
                    (audio_waveform.shape[0],),
                    audio_waveform.shape[1],
                    dtype=torch.long,
                    device=audio_waveform.device,
                )
            else:
                lengths = audio_waveform_lengths.to(audio_waveform.device).long()
            valid_rows = (lengths > 0).nonzero(as_tuple=True)[0]
            if len(valid_rows) == 0:
                return None, None

            valid_lengths = lengths[valid_rows]
            max_len = int(valid_lengths.max().item())
            wav = audio_waveform[valid_rows, :max_len].to(ced_device)
            valid_lengths = valid_lengths.to(ced_device)
            time = torch.arange(max_len, device=ced_device).unsqueeze(0)
            wav = wav.masked_fill(time >= valid_lengths.unsqueeze(1), 0.0)

            with torch.no_grad():
                device_type = "cuda" if wav.device.type == "cuda" else "cpu"
                with torch.autocast(device_type=device_type, enabled=False):
                    ced_output = self.ced_model(wav.float())
                hs = ced_output.hidden_states
                if len(hs) <= 7:
                    return None, None
                proc_dtype = next(self.ced_processor.parameters()).dtype
                f4 = hs[3].to(proc_dtype)
                f8 = hs[7].to(proc_dtype)
                fl = hs[-1].to(proc_dtype)

            ced_valid_lengths = (valid_lengths // 2560).to(torch.long)
            features = self.ced_processor(f4, f8, fl, valid_lengths=ced_valid_lengths)
            if features is None:
                return None, None
            return features, valid_rows.to(audio_waveform.device)
        except Exception as e:
            logger.warning(f"CED feature extraction failed: {e}. Skipping CED for this batch.")
            return None, None

    def _prepare_inputs_embeds(
        self,
        input_ids,
        inputs_embeds,
        input_features,
        feature_attention_mask,
        audio_feature_lengths,
        audio_waveform,
        audio_waveform_lengths,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_features = None
        audio_output_lengths = None
        valid_rows = None
        if input_features is not None:
            audio_features, audio_output_lengths, audio_feature_lengths, valid_rows = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                return_lengths=True,
            )
            if audio_features is None:
                return inputs_embeds, audio_feature_lengths
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            token_audio_mask = input_ids == self.config.audio_token_id
            self._scatter_audio_features_by_sample(
                inputs_embeds=inputs_embeds,
                audio_features=audio_features,
                audio_lengths=audio_output_lengths,
                audio_mask=token_audio_mask,
                valid_rows=valid_rows,
            )
            inputs_embeds = self._add_ced_features(
                inputs_embeds,
                input_ids,
                token_audio_mask,
                audio_waveform,
                audio_waveform_lengths,
                audio_feature_lengths,
                audio_output_lengths,
                valid_rows,
            )
        elif feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        return inputs_embeds, audio_feature_lengths

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        audio_waveform: Optional[torch.FloatTensor] = None,
        audio_waveform_lengths: Optional[torch.LongTensor] = None,
        audio_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        del pixel_values, pixel_values_videos, audio_mask, rope_deltas

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds, audio_feature_lengths = self._prepare_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            audio_waveform=audio_waveform,
            audio_waveform_lengths=audio_waveform_lengths,
        )

        if attention_mask is not None and position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if past_key_values_length == 0 or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                self.rope_deltas = rope_deltas - delta0
            else:
                batch_size, seq_length = input_ids.shape
                delta = (past_key_values_length + self.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        audio_waveform=None,
        audio_waveform_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_waveform=audio_waveform,
            audio_waveform_lengths=audio_waveform_lengths,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        model_inputs["position_ids"] = None
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None
            model_inputs["feature_attention_mask"] = None
            model_inputs["audio_waveform"] = None
            model_inputs["audio_waveform_lengths"] = None
        return model_inputs
