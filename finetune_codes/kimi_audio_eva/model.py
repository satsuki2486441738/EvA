# kimi_audio_eva/model.py
import os
import argparse
from contextlib import nullcontext
from typing import Optional, List
import torch
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from kimia_infer.models.tokenizer.ced_base.modeling_ced import CedEncoder

from .modeling_kimia import MoonshotKimiaForCausalLM
from transformers.utils import logging as hf_logging
logger = hf_logging.get_logger(__name__)

def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0, min_p: float = 0.0, filter_value=-float("inf")):
    """
    对 logits 做 top-k / top-p / min-p 过滤（逐样本）。
    - top-k: 只保留概率前 k
    - top-p: nucleus sampling
    - min-p: 过滤掉小于阈值的 token 概率
    """
    # top-k
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < kth_vals, torch.tensor(filter_value, device=logits.device, dtype=logits.dtype), logits)

    # top-p
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_mask = cumulative_probs > top_p
        # shift mask right to keep the first token above the threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = torch.scatter(
            torch.zeros_like(sorted_mask, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    # min-p（在 top-p 之后做）
    if min_p and min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_p, filter_value)

    return logits


class KimiAudioModel(MoonshotKimiaForCausalLM):
    """for training, containing whisper and ced models"""
    def __init__(self, config):
        super().__init__(config)
        # 训练态：外部装载
        self.whisper_model = None
        self.ced_model = None
        # 消融开关：是否在前向传播中使用CED特征
        self.use_ced_in_forward = True

    @classmethod
    def init_from_pretrained(cls, model_name_or_path: str, model_load_kwargs: dict):
        print(f"Loading KimiAudioModel from: {model_name_or_path}")
        if os.path.exists(model_name_or_path):
            cache_path = model_name_or_path
        else:
            print(f"'{model_name_or_path}' is not a local path. Downloading from Hugging Face Hub...")
            cache_path = snapshot_download(model_name_or_path)

        # 1) 加载 LLM 主体
        print("Loading base Language Model (MoonshotKimiaForCausalLM)...")
        model = super(KimiAudioModel, cls).from_pretrained(
            cache_path,
            trust_remote_code=True,
            **model_load_kwargs,
        )
        print("Base Language Model loaded successfully.")

        # 2) 加载外部音频编码器
        whisper_path = os.path.join(cache_path, "whisper-large-v3")
        ced_path = os.path.join(cache_path, "ced-base")
        if not os.path.exists(whisper_path):
            raise FileNotFoundError(f"Whisper model directory not found at: {whisper_path}")
        if not os.path.exists(ced_path):
            raise FileNotFoundError(f"CED model directory not found at: {ced_path}")

        model.whisper_model = WhisperEncoder(whisper_path, mel_batch_size=20)
        model.ced_model = CedEncoder(ced_path)

        # 3) 精度策略：Whisper -> BF16, CED -> FP32（满足 CED 中 LN 的 FP32 约束）
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        try:
            model.whisper_model.to(device=dev, dtype=torch.bfloat16).eval()
        except Exception:
            # 个别实现不支持直接切 dtype，退化成仅 to(device) + eval
            model.whisper_model.to(device=dev).eval()
        # CED 始终 FP32
        model.ced_model.to(device=dev, dtype=torch.float32).eval()

        print("KimiAudioModel successfully assembled with Whisper and CED components.")
        return model

    @staticmethod
    def export_model(model_or_path, output_dir, src_submodules: Optional[object] = None):
        from .export_utils import export_model as _export_model
        if isinstance(model_or_path, str):
            print(f"Loading model from {model_or_path}")
            kimiaudio = KimiAudioModel.from_pretrained(model_or_path)
        else:
            kimiaudio = model_or_path
        _export_model(kimiaudio, output_dir, src_submodules=src_submodules)

    def forward(
        self,
        audio_input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        waveform: Optional[torch.FloatTensor] = None,
        waveform_lengths: Optional[torch.LongTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        训练前向：
        - Whisper 编码：BF16（autocast）
        - CED 编码：强制 FP32（禁用 autocast），满足 LayerNorm FP32 约束
        - 主干：BF16（autocast）
        """
        whisper_feats = None
        ced_feats_tuple = None

        if waveform is not None and waveform.numel() > 0:
            with torch.no_grad():
                # 将波形放到与模型一致的设备
                dev = next(self.parameters()).device
                waveform = waveform.to(dev)

                # Whisper feature extraction（允许 BF16）
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        whisper_output = self.whisper_model(waveform)
                except Exception:
                    whisper_output = self.whisper_model(waveform)

                whisper_feats = whisper_output.reshape(
                    whisper_output.shape[0],
                    int(whisper_output.shape[1] // 4),
                    whisper_output.shape[2] * 4,
                ).to(torch.bfloat16)

                # CED feature extraction（强制 FP32：禁用 autocast）
                # 消融开关：use_ced_in_forward=False 时跳过CED特征提取
                if self.use_ced_in_forward:
                    try:
                        with torch.autocast(device_type="cuda", enabled=False):
                            ced_output = self.ced_model(waveform.float())
                    except Exception:
                        ced_output = self.ced_model(waveform.float())

                    all_ced_hidden_states = ced_output.hidden_states
                    if len(all_ced_hidden_states) > 7:
                        # 先保持 FP32，随后传给主干前转 BF16（主干/投影通常是 BF16）
                        ced_feat_4 = all_ced_hidden_states[3].to(torch.bfloat16)
                        ced_feat_8 = all_ced_hidden_states[7].to(torch.bfloat16)
                        ced_feat_last = all_ced_hidden_states[-1].to(torch.bfloat16)
                        ced_feats_tuple = (ced_feat_4, ced_feat_8, ced_feat_last)
                    else:
                        logger.warning(
                            f"CED model has only {len(all_ced_hidden_states)} layers. Cannot extract features from layers 4 and 8."
                        )

        # 主干 BF16 计算
        use_cuda_autocast = torch.cuda.is_available()
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_autocast else nullcontext()

        # 从 waveform_lengths 计算 CED 池化后的有效时间步数
        # CED encoder: hop_size=160 samples * patch_stride=16 mel frames → 每 2560 samples 一个池化时间步
        ced_valid_lengths = None
        if waveform_lengths is not None and self.use_ced_in_forward:
            ced_valid_lengths = (waveform_lengths.to(torch.long) // 2560).clamp(min=0)

        with autocast_ctx:
            return super().forward(
                audio_input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=whisper_feats,
                ced_input_feature=ced_feats_tuple,
                ced_valid_lengths=ced_valid_lengths,
                is_continuous_mask=is_continuous_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                generation_mode=generation_mode,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        *,
        text_input_ids: torch.LongTensor,
        audio_input_ids: torch.LongTensor = None,
        is_continuous_mask: torch.Tensor = None,
        waveform: torch.FloatTensor = None,
        attention_mask: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        kimia_processor=None,   # 用于取 blank/eos 等（可选）
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        min_p: float = 0.0,
        eos_token_id: int = None,
        pad_token_id: int = None,
        use_cache: bool = True,
        **unused,
    ):
        """
        自定义生成（文本续写）：首步支持多模态，续步只采样文本（音频流写 blank）。
        """
        device = text_input_ids.device
        B, _ = text_input_ids.shape
        eos = eos_token_id
        if eos is None and hasattr(self.config, "eos_token_id"):
            eos = self.config.eos_token_id
        if eos is None and kimia_processor is not None:
            eos = getattr(kimia_processor.tokenizer, "eos_token_id", None)

        pad = pad_token_id
        if pad is None and hasattr(self.config, "pad_token_id"):
            pad = self.config.pad_token_id
        if pad is None and kimia_processor is not None:
            pad = getattr(kimia_processor, "pad_token_id", None)
        if pad is None:
            pad = 0

        # attention mask 兜底
        if attention_mask is None:
            if audio_input_ids is not None:
                nonpad_audio = (audio_input_ids != pad)
            else:
                nonpad_audio = torch.zeros_like(text_input_ids, dtype=torch.bool)

            if text_input_ids is not None:
                nonpad_text = (text_input_ids != pad)
            else:
                nonpad_text = torch.zeros_like(audio_input_ids, dtype=torch.bool)

            attention_mask = (nonpad_audio | nonpad_text)

        past_key_values = None
        cur_text = text_input_ids
        cur_audio = audio_input_ids
        cur_mask = is_continuous_mask
        cur_wave = waveform

        generated = []
        finished_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            outputs = self(
                audio_input_ids=cur_audio,
                text_input_ids=cur_text,
                waveform=cur_wave,
                is_continuous_mask=cur_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            text_logits = outputs.logits[:, -1, :]  # [B, V]

            # 已完成样本屏蔽
            if eos is not None:
                text_logits = text_logits.masked_fill(finished_mask.unsqueeze(-1), float("-inf"))
                text_logits[:, eos] = torch.where(
                    finished_mask,
                    torch.tensor(0.0, device=device, dtype=text_logits.dtype),
                    text_logits[:, eos],
                )

            # 温度缩放 + 过滤
            if temperature and temperature > 0:
                text_logits = text_logits / float(temperature)
            text_logits = _top_k_top_p_filtering(text_logits, top_k=top_k, top_p=top_p, min_p=min_p)

            # 采样/贪心
            if do_sample:
                probs = torch.softmax(text_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # [B,1]
            else:
                next_tokens = torch.argmax(text_logits, dim=-1, keepdim=True)

            # 更新完成标记
            if eos is not None:
                finished_mask |= (next_tokens.squeeze(-1) == eos)
            generated.append(next_tokens)

            if torch.all(finished_mask):
                break

            # 续步：只喂文本；音频给 blank 占位；不再需要 waveform/连续掩码
            cur_text = next_tokens
            blank = kimia_processor.extra.kimia_text_blank if kimia_processor is not None else 0
            cur_audio = torch.full_like(cur_text, blank)
            cur_mask  = torch.zeros_like(cur_text, dtype=torch.bool)
            cur_wave  = None

            # attention_mask / position_ids 递增一位
            attention_mask = torch.cat(
                [attention_mask,
                torch.ones(attention_mask.size(0), 1, device=device, dtype=torch.bool)],
                dim=1
            )
            pos = attention_mask.long().cumsum(-1) - 1
            pos.masked_fill_(attention_mask == 0, 0)
            position_ids = pos[:, -1:]

        if len(generated) == 0:
            return text_input_ids
        new_seq = torch.cat(generated, dim=1)  # [B, L_new]
        return torch.cat([text_input_ids, new_seq], dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "separate"], default="separate")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    new_initial_model: bool = False

    if args.action == "init_from_pretrained":
        # init model and save
        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})
        if new_initial_model:
            model._initialize_newly_added_modules()
            KimiAudioModel.export_model(model, args.output_dir)
    elif args.action == "separate":
        # separate pretrained model into three submodel
        KimiAudioModel.export_model(args.model_name, args.output_dir)
