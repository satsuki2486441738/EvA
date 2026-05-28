# kimia_infer/api/kimia.py
import os

import tqdm
import torch
from loguru import logger
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler


class KimiAudio(object):
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info("Loading Kimi-Audio main model")

        if os.path.exists(model_path):
            cache_path = model_path
        else:
            cache_path = snapshot_download(model_path)

        logger.info(f"Looking for resources in {cache_path}")
        logger.info("Loading causal LM (with KimiAudio head)")
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()

        model_config = self.alm.config
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path,
            kimia_token_offset=self.kimia_token_offset,
            kimia_text_audiodelaytokens=getattr(model_config, 'kimia_mimo_audiodelaytokens', 5),
        )

        if load_detokenizer:
            logger.info("Loading detokenizer (first run may compile extensions)")
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            # in this case, you're not allowed to generate audio(wav)
            self.detokenizer = None

        self.prompt_manager._maybe_load_text()
        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor = None,
        max_new_tokens: int = 50,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        whisper_input_feature: torch.Tensor = None,
        ced_input_feature: list[torch.Tensor] = None,
    ):
        """
        文本专用采样循环。返回 ([], text_tokens) 保持调用侧接口兼容。
        """
        max_new_tokens = max(1, int(max_new_tokens))

        sampler = KimiASampler(
            audio_top_k=0, audio_temperature=0.0,
            audio_repetition_penalty=1.0, audio_repetition_window_size=64,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        device = next(self.alm.parameters()).device
        text_previous_tokens = torch.empty((max_new_tokens,), dtype=torch.long, device=device)

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids  = text_input_ids.clone()
        decoder_position_ids    = torch.arange(0, decoder_input_audio_ids.shape[1], device=device).unsqueeze(0).long()
        decoder_input_whisper_feature = whisper_input_feature
        decoder_input_ced_feature     = ced_input_feature
        decoder_is_continuous_mask    = is_continuous_mask
        past_key_values = None

        last_position_id  = decoder_input_audio_ids.shape[1] - 1
        valid_text_length = 0

        for i in tqdm.tqdm(range(max_new_tokens), desc="Generating tokens", disable=False):
            out = self.alm.forward(
                audio_input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                ced_input_feature=decoder_input_ced_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            text_logits = out.logits
            past_key_values = out.past_key_values

            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=(text_previous_tokens[:i] if i > 0 else None)
            )

            if int(next_token_text) == self.extra_tokens.kimia_text_eos:
                return [], text_previous_tokens[:valid_text_length].detach().cpu().tolist()

            text_previous_tokens[i:i+1] = next_token_text
            valid_text_length += 1

            decoder_input_audio_ids = torch.full_like(next_token_text, self.extra_tokens.kimia_text_blank).unsqueeze(1)
            decoder_input_text_ids  = next_token_text.unsqueeze(1)
            decoder_position_ids    = torch.full((1, 1), last_position_id + 1, device=device, dtype=torch.long)
            last_position_id += 1

            decoder_input_whisper_feature = None
            decoder_input_ced_feature     = None
            decoder_is_continuous_mask    = None

        return [], text_previous_tokens[:valid_text_length].detach().cpu().tolist()

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
    ):
        """
        output_type: 保留参数以兼容调用方，始终执行文本生成。
        """
        history = self.prompt_manager.get_prompt(chats, output_type="text")

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
        whisper_features = history.continuous_feature
        ced_features = history.ced_hidden_states

        if max_new_tokens == -1:
            max_new_tokens = max(1, 7500 - audio_input_ids.shape[1])

        device = next(self.alm.parameters()).device
        audio_input_ids = audio_input_ids.to(device)
        text_input_ids  = text_input_ids.to(device)
        is_continuous_mask = is_continuous_mask.to(device)
        whisper_features, ced_features = self._normalize_features(whisper_features, ced_features, device)

        _, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            whisper_input_feature=whisper_features,
            ced_input_feature=ced_features,
        )

        generated_text_tokens = [t for t in generated_text_tokens if t < self.kimia_token_offset]
        generated_text = self.detokenize_text(generated_text_tokens)

        return None, generated_text

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []

        device = next(self.alm.parameters()).device
        audio_tokens = audio_tokens.to(device).long()

        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)

    @torch.inference_mode()
    def _generate_loop_text_only(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        whisper_input_feature: torch.Tensor = None,
        ced_input_feature: list[torch.Tensor] = None,
    ):
        """
        纯文本快速路径：首步用到多模态特征，后续仅续写文本，并用 blank 音频 token 占位。
        """
        device = next(self.alm.parameters()).device
        sampler = KimiASampler(
            audio_top_k=0, audio_temperature=0.0,
            audio_repetition_penalty=1.0, audio_repetition_window_size=64,
            text_top_k=text_top_k, text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty, text_repetition_window_size=text_repetition_window_size
        )

        last_pos = audio_input_ids.size(1) - 1
        decoder_audio_ids = audio_input_ids.to(device)
        decoder_text_ids  = text_input_ids.to(device)
        decoder_pos_ids   = torch.arange(0, decoder_audio_ids.shape[1], device=device).unsqueeze(0).long()
        decoder_cont_mask = is_continuous_mask.to(device) if is_continuous_mask is not None else None

        # 允许首步带上特征
        whisper_feat = whisper_input_feature
        ced_feat     = ced_input_feature

        text_prev = []
        past_kv = None

        for i in range(max_new_tokens):
            out = self.alm.forward(
                audio_input_ids=decoder_audio_ids,
                text_input_ids=decoder_text_ids,
                whisper_input_feature=whisper_feat,
                ced_input_feature=ced_feat,
                is_continuous_mask=decoder_cont_mask,
                position_ids=decoder_pos_ids,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            text_logits = out.logits
            past_kv = out.past_key_values

            next_token_text = sampler.sample_text_logits(
                text_logits,
                recent_tokens=(torch.tensor(text_prev, device=device) if len(text_prev) > 0 else None),
            )
            t = next_token_text.item()
            text_prev.append(t)

            if t == self.extra_tokens.kimia_text_eos:
                break

            # 续步只喂新文本；音频给 blank 占位
            decoder_audio_ids = torch.full_like(next_token_text, self.extra_tokens.kimia_text_blank).unsqueeze(1)
            decoder_text_ids  = next_token_text.unsqueeze(1)
            decoder_pos_ids   = torch.full((1, 1), last_pos + 1, device=device, dtype=torch.long)
            last_pos += 1

            # 续步后不再需要连续特征
            whisper_feat = None
            ced_feat     = None
            decoder_cont_mask = None

        return [], text_prev

    def _normalize_features(self, whisper_features, ced_features, device):
        # whisper: Tensor or list[Tensor] -> Tensor[B,T,D]
        if whisper_features is None:
            whisper = None
        elif isinstance(whisper_features, torch.Tensor):
            whisper = whisper_features.to(device, dtype=torch.bfloat16)
        elif isinstance(whisper_features, (list, tuple)):
            parts = [t.to(device, dtype=torch.bfloat16) for t in whisper_features if isinstance(t, torch.Tensor)]
            if len(parts) == 0:
                whisper = None
            else:
                # 要求 batch/hidden 相同，按时间维拼接
                whisper = torch.cat(parts, dim=1)
        else:
            whisper = None

        # ced:
        #   (a) (f4,f8,flast) -> 保留
        #   (b) [all_hidden_states] -> 抽取第4/8/last层，兼容旧 prompt_manager 输出
        #   (c) list/tuple of per-chunk tuples -> 分别拼接后再组成三元组
        def _looks_like_hidden_state_stack(seq):
            return (
                isinstance(seq, (list, tuple))
                and len(seq) > 7
                and all(isinstance(t, torch.Tensor) for t in seq)
            )

        def _as_tuple3(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                # case (a)
                if len(x) == 3 and all(isinstance(t, torch.Tensor) for t in x):
                    return tuple(t.to(device, dtype=torch.bfloat16) for t in x)
                # case (b): a single audio item stored as [all_hidden_states]
                if len(x) == 1 and _looks_like_hidden_state_stack(x[0]):
                    hs = x[0]
                    return (
                        hs[3].to(device, dtype=torch.bfloat16),
                        hs[7].to(device, dtype=torch.bfloat16),
                        hs[-1].to(device, dtype=torch.bfloat16),
                    )
                # case (b): all_hidden_states passed directly
                if _looks_like_hidden_state_stack(x):
                    return (
                        x[3].to(device, dtype=torch.bfloat16),
                        x[7].to(device, dtype=torch.bfloat16),
                        x[-1].to(device, dtype=torch.bfloat16),
                    )
                # case (c)
                if all(isinstance(t, (list, tuple)) and len(t) == 3 for t in x):
                    f4  = [t[0].to(device, dtype=torch.bfloat16) for t in x]
                    f8  = [t[1].to(device, dtype=torch.bfloat16) for t in x]
                    fl  = [t[2].to(device, dtype=torch.bfloat16) for t in x]
                    return (torch.cat(f4, dim=1), torch.cat(f8, dim=1), torch.cat(fl, dim=1))
            if isinstance(x, torch.Tensor):
                # 兼容极端：只给了最后层，则重复三份兜底（不建议，但不至于崩）
                return (x.to(device, dtype=torch.bfloat16),) * 3
            return None

        ced = _as_tuple3(ced_features)
        return whisper, ced