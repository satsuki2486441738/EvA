# kimia_infer/api/prompt_manager.py
from typing import List, Dict
import os
import hashlib
import json as pyjson

import librosa
import torch, torchaudio
from contextlib import nullcontext
from loguru import logger
from transformers import AutoTokenizer, AutoConfig

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimia_infer.models.tokenizer.ced_base.modeling_ced import CedEncoder

from kimia_infer.utils.data import KimiAContent
from kimia_infer.utils.special_tokens import instantiate_extra_tokens


class KimiAPromptManager:
    """
    主要改动（接口不变）：
    1) 懒加载 Whisper/CED/Text tokenizer（init 不再立即占用 GPU/时间；首次用到时再加载）。
    2) _tokenize_audio：inference_mode + autocast(bf16/fp16)；增加内存/可选落盘缓存（通过环境变量开启）。
    3) tokenize_message：修复 audio_path 作用域；需要用到 extra_tokens 时自动加载文本分词器。
    4) extract_*：更快的 torchaudio I/O（必要时自动重采样到 16k，单声道）。
    """

    def __init__(self, model_path: str, kimia_token_offset: int, kimia_text_audiodelaytokens: int):
        logger.info(f"Looking for resources in {model_path}")

        # 记录必要信息
        self._model_path = model_path
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.kimia_token_offset = kimia_token_offset

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # —— 懒加载占位（首次使用时再真正加载）——
        self.whisper_model = None
        self.ced_model = None
        self.text_tokenizer = None
        self.extra_tokens = None

        # Audio tokenizer 仍立即加载（这是预标注 audio_tokens 的主通路）
        # 保持官方推荐的 float32 精度，确保量化 token 与官方一致
        logger.info(f"Loading audio tokenizer")
        self.audio_tokenizer = Glm4Tokenizer("zai-org/glm-4-voice-tokenizer").to(device)

        # 可选：落盘缓存目录（通过环境变量启用；不改任何对外接口）
        self._cache_dir = os.environ.get("KIMIA_AUDIO_TOKEN_CACHE", "")
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
        self._audio_cache = {}

    # -------------------
    # 懒加载辅助
    # -------------------
    def _maybe_load_text(self):
        if self.text_tokenizer is None or self.extra_tokens is None:
            logger.info("Loading text tokenizer")
            mp = self._model_path
            if os.path.exists(mp) and os.path.exists(os.path.join(mp, "tokenizer_config.json")):
                self.text_tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
            else:
                logger.info(f"Can not find text tokenizer in {mp}, Loading default text tokenizer from moonshotai/Kimi-Audio-7B-Instruct")
                self.text_tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-Audio-7B-Instruct", trust_remote_code=True)
            self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

    def _maybe_load_whisper(self):
        if self.whisper_model is None:
            logger.info("Lazy-loading Whisper model")
            self.whisper_model = WhisperEncoder(os.path.join(self._model_path, "whisper-large-v3"), mel_batch_size=20)
            self.whisper_model = self.whisper_model.to(torch.cuda.current_device())
            try:
                if self._amp_dtype is torch.bfloat16:
                    self.whisper_model = self.whisper_model.bfloat16()
                else:
                    self.whisper_model = self.whisper_model.half()
            except Exception:
                pass
            self.whisper_model.eval()

    def _maybe_load_ced(self):
        if self.ced_model is None:
            logger.info("Lazy-loading CED model")
            self.ced_model = CedEncoder(os.path.join(self._model_path, "ced-base"))
            self.ced_model = self.ced_model.to(torch.cuda.current_device()).float()
            self.ced_model.eval()

    # -------------------
    # 文本/音频 Tokenize
    # -------------------
    def _tokenize_text(self, text):
        if text is None:
            return None
        self._maybe_load_text()
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def _cache_key(self, wav_path: str):
        try:
            st = os.stat(wav_path)
            return f"{wav_path}|{st.st_mtime_ns}|{st.st_size}"
        except Exception:
            return wav_path

    def _cache_file(self, key: str):
        if not self._cache_dir:
            return None
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return os.path.join(self._cache_dir, f"{h}.json")

    @torch.inference_mode()
    def _tokenize_audio(self, wav_path):
        # 内存/落盘缓存
        key = self._cache_key(wav_path)
        if key in self._audio_cache:
            return self._audio_cache[key]
        cf = self._cache_file(key)
        if cf and os.path.isfile(cf):
            try:
                with open(cf, "r") as f:
                    obj = pyjson.load(f)
                    self._audio_cache[key] = obj
                    return obj
            except Exception:
                pass

        # 推理：audio tokenizer 保持 float32（官方推荐精度），不使用 autocast
        with nullcontext():
            wav_tokens = self.audio_tokenizer.tokenize(audio_path=wav_path)
        wav_tokens = wav_tokens + self.kimia_token_offset
        wav_tokens_list = wav_tokens.squeeze(0).detach().cpu().tolist()

        # 写回缓存
        self._audio_cache[key] = wav_tokens_list
        if cf:
            try:
                with open(cf, "w") as f:
                    pyjson.dump(wav_tokens_list, f)
            except Exception:
                pass
        return wav_tokens_list

    # -------------------
    # 更快的音频 I/O（torchaudio）
    # -------------------
    def _load_wav_16k(self, wav: torch.Tensor | str):
        if isinstance(wav, str):
            # 更快且是 C 实现；必要时自动重采样
            wav_tensor, sr = torchaudio.load(wav)  # [C, T]
            if sr != 16000:
                wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
            wav_tensor = wav_tensor.mean(0, keepdim=True)  # 转单声道 [1, T]
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav
        else:
            raise ValueError(f"Invalid wav type: {type(wav)}")
        return wav_tensor

    # -------------------
    # Whisper / CED 特征
    # -------------------
    def extract_whisper_feat(self, wav: torch.Tensor | str):
        self._maybe_load_whisper()
        wav_tensor = self._load_wav_16k(wav)
        wav_tensor = wav_tensor.to(torch.cuda.current_device(), non_blocking=True)
        with torch.inference_mode():
            continous_feature = self.whisper_model.tokenize_waveform(wav_tensor)
            continous_feature = continous_feature.reshape(
                continous_feature.shape[0],
                int(continous_feature.shape[1] // 4),
                continous_feature.shape[2] * 4,
            )
        return continous_feature

    def extract_ced_feat(self, wav: torch.Tensor | str) -> List[torch.Tensor]:
        self._maybe_load_ced()
        wav_tensor = self._load_wav_16k(wav).to(torch.cuda.current_device(), non_blocking=True)
        # 显式 FP32 前向，防止外层开启了 autocast 影响到 CED
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            output = self.ced_model(wav_tensor.float())
        hidden_states = output.hidden_states
        if len(hidden_states) <= 7:
            raise RuntimeError(
                f"CED model returned only {len(hidden_states)} hidden states; "
                "EvA requires layers 4, 8, and last."
            )
        return (hidden_states[3], hidden_states[7], hidden_states[-1])

    # -------------------
    # 高层封装（维持原接口）
    # -------------------
    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        extract_whisper_feature=False,
        extract_ced_feature=False,
        output_type: str = "text",
    ):
        # 需要使用 extra_tokens，因此确保文本分词器可用
        self._maybe_load_text()

        kimia_content_msg = KimiAContent()

        role = message["role"]
        has_loss = role == "assistant"

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))

            if role == "assistant":
                # eos for text stream
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        elif message["message_type"] == "audio":
            # —— 修复：无论是否已有 audio_tokens，都先取出路径 —— #
            audio_path = message["content"]

            if "audio_tokens" in message:
                speech_tokens = message["audio_tokens"]
            else:
                speech_tokens = self._tokenize_audio(audio_path)

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            # EOS for audio stream
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss)
            kimia_content_msg.text_extend([self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2))

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ctd_id)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if extract_whisper_feature:
                whisper_feature = self.extract_whisper_feat(audio_path)
                kimia_content_msg.continuous_feature.append(whisper_feature)
            if extract_ced_feature:
                ced_hidden_states = self.extract_ced_feat(audio_path)
                kimia_content_msg.ced_hidden_states.append(ced_hidden_states)

        elif message["message_type"] == "audio-text":
            audio_path, text = message["content"]
            speech_tokens = self._tokenize_audio(audio_path)
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * self.kimia_text_audiodelaytokens)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
            kimia_content_msg.text_extend(text_tokens)
            text_pad_tokens = (self.kimia_text_audiodelaytokens + len(speech_tokens) - len(text_tokens)) * [
                self.extra_tokens.kimia_text_blank
            ]
            kimia_content_msg.text_extend(text_pad_tokens)

        elif message["message_type"] is None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert kimia_content_msg.is_valid(), f"kimia_content_msg is not valid: {kimia_content_msg}"
        return kimia_content_msg

    def get_prompt(
        self, messages: List[Dict], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> KimiAContent:
        """
        messages: List[Dict]
        messages[i] = {
            "role": "user" | "assistant" | "system",
            "content": str
        }
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            if previous_role is None:
                tokenize_role = True
            else:
                tokenize_role = (message["role"] != previous_role)

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                # 保持原默认：需要时才会懒加载并抽取
                extract_whisper_feature=True,
                extract_ced_feature=True,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                message={
                    "role": "assistant",
                    "message_type": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )
            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]
        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg
