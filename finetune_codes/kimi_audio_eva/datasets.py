# kimi_audio_eva/datasets.py
from torch.utils.data import Dataset
from functools import lru_cache
import os
import torch, torchaudio
from typing import Dict, List
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging

logger = logging.getLogger(__name__)

MIN_AUDIO_SAMPLES = 1024  # 64ms @ 16kHz，低于此值 CED STFT 不可靠
MIN_AUDIO_DURATION = MIN_AUDIO_SAMPLES / 16000  # 换算为秒，与原始采样率无关


def _check_audio_ok(path: str) -> bool:
    """检查音频文件是否可用。文件不存在、损坏或过短时返回 False。"""
    if not os.path.exists(path):
        logger.warning(f"Audio file not found: {path}")
        return False
    try:
        info = torchaudio.info(path)
        if info.num_frames / info.sample_rate < MIN_AUDIO_DURATION:
            logger.warning(f"Audio too short ({info.num_frames / info.sample_rate:.3f}s < {MIN_AUDIO_DURATION:.3f}s): {path}")
            return False
    except Exception as e:
        logger.warning(f"Cannot read audio metadata {path}: {e}")
        return False
    return True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data_list, text_tokenizer, max_len: int, kimia_token_offset: int, skip_audio_check: bool = False):
        super(LazySupervisedDataset, self).__init__()
        self.max_len = max_len

        # 过滤 audio_tokens 为 None 或长度过短的坏样本
        MIN_AUDIO_TOKENS = 8  # 避免whisper编码长度与audio token数量不匹配
        valid_data = [
            s for s in raw_data_list
            if all(
                msg.get("audio_tokens") is not None and len(msg.get("audio_tokens", [])) >= MIN_AUDIO_TOKENS
                for msg in s.get("conversation", [])
                if msg.get("message_type") == "audio"
            )
        ]
        n_bad = len(raw_data_list) - len(valid_data)
        if n_bad:
            print(f"Warning: filtered out {n_bad} samples (audio_tokens=None or len<{MIN_AUDIO_TOKENS})")

        # 过滤音频文件不存在、损坏或过短的样本，并显式执行当前训练假设：每条样本只含一段音频。
        def _get_audio_paths(item):
            return [msg["content"] for msg in item.get("conversation", [])
                    if msg.get("message_type") == "audio" and msg.get("content")]

        n_before = len(valid_data)
        valid_data = [s for s in valid_data if len(_get_audio_paths(s)) == 1]
        n_bad_count = n_before - len(valid_data)
        if n_bad_count:
            print(f"Warning: filtered out {n_bad_count} samples (expected exactly one audio message)")

        if skip_audio_check:
            print("Warning: skip_audio_check=True; audio files will be validated when loaded")
        else:
            n_before = len(valid_data)
            valid_data = [s for s in valid_data
                          if all(_check_audio_ok(p) for p in _get_audio_paths(s))]
            n_bad_audio = n_before - len(valid_data)
            if n_bad_audio:
                print(f"Warning: filtered out {n_bad_audio} samples (audio file missing/corrupt/short)")

        print(f"Loading text tokenizer")
        self.text_tokenizer = text_tokenizer

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        self.pad_token = self.extra_tokens.pad
        self.kimia_token_offset = kimia_token_offset

        n_before = len(valid_data)
        valid_data = [s for s in valid_data if self._conversation_len_ok(s)]
        n_bad_len = n_before - len(valid_data)
        if n_bad_len:
            print(f"Warning: filtered out {n_bad_len} samples (tokenized length > max_len={self.max_len})")

        print("There are {} samples in the dataset".format(len(valid_data)))
        self.raw_data = valid_data

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def _load_wav_once(self, wav_path):
        with torch.no_grad():
            waveform, sample_rate = torchaudio.load(wav_path, normalize=True)  # [C, T] or [T]

            # 统一成 [C, T]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # -> [1, T]
            # 任意多声道都合成单声道
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # -> [1, T]

            # 重采样到 16k
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)  # 期望 [1, T']

            # 返回 1D [T]
            return waveform.squeeze(0).contiguous()

    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def _conversation_len_ok(self, sample) -> bool:
        if not self.max_len or self.max_len <= 0:
            return True
        task_type = sample.get("task_type")
        output_type = "text" if task_type == "understanding" else "both"
        tokenized = self.tokenize_conversation(
            sample.get("conversation", []),
            output_type=output_type,
            add_assistant_start_msg=False,
        )
        return len(tokenized.audio_token_ids) <= self.max_len and len(tokenized.text_token_ids) <= self.max_len

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        output_type: str = "text",
    ):
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
                # 文本流 EOS；音频流补一个 blank 对齐且不计 loss
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)  # eos for text stream
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        elif message["message_type"] == "audio":
            speech_tokens = message["audio_tokens"]
            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss)
            kimia_content_msg.text_extend([self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2))

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ctd_id)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            # 不在这里加载波形；仅保存路径，由 __getitem__/collate 再处理
            kimia_content_msg.continuous_feature.append(message["content"])  # 路径

        elif message["message_type"] is None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert kimia_content_msg.is_valid(), f"kimia_content_msg is not valid: {kimia_content_msg}"
        return kimia_content_msg

    def tokenize_conversation(
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
        previous_role = None

        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            tokenize_role = (previous_role is None) or (message["role"] != previous_role)
            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                has_ct_token = messages[msg_idx + 1]["role"] != message["role"]
                has_msg_end_token = messages[msg_idx + 1]["role"] != message["role"]

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                message={"role": "assistant", "message_type": None},
                tokenize_role=True, has_ct_token=False, has_msg_end_token=False,
            )
            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]
        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_type = self.raw_data[idx]["task_type"]
        conversation = self.raw_data[idx]["conversation"]
        output_type = "text" if task_type == "understanding" else "both"
        tokenized_conversation = self.tokenize_conversation(conversation, output_type=output_type, add_assistant_start_msg=False)

        audio_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, text_token_loss_mask = tokenized_conversation.to_tensor()

        audio_paths = tokenized_conversation.continuous_feature
        if len(audio_paths) != 1:
            raise ValueError(f"Expected exactly one audio path per sample, got {len(audio_paths)} at idx={idx}")
        wav_path = audio_paths[0]
        waveform = self._load_wav_once(wav_path)

        # 右移一位，构造 labels 与 loss mask
        audio_labels = torch.cat((audio_input_ids[:, 1:], audio_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        text_labels = torch.cat((text_input_ids[:, 1:], text_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        audio_loss_mask = torch.cat((audio_token_loss_mask[:, 1:], audio_token_loss_mask.new_full((1, 1), False)), dim=1)
        text_loss_mask = torch.cat((text_token_loss_mask[:, 1:], text_token_loss_mask.new_full((1, 1), False)), dim=1)

        return dict(
            audio_input_ids=audio_input_ids.squeeze(0),
            text_input_ids=text_input_ids.squeeze(0),
            is_continuous_mask=is_continuous_mask.squeeze(0),
            waveform=waveform,
            labels=(
                audio_labels.squeeze(0),
                text_labels.squeeze(0),
                audio_loss_mask.squeeze(0),
                text_loss_mask.squeeze(0),
            ),
        )

    @staticmethod
    def _left_pad_1d(t: torch.Tensor, target_len: int, pad_value):
        L = t.size(0)
        if L == target_len:
            return t
        pad = (target_len - L, 0)  # (left, right)
        return F.pad(t, pad, value=pad_value)

    @staticmethod
    def _left_pad_bool(t: torch.Tensor, target_len: int, pad_value: bool):
        L = t.size(0)
        if L == target_len:
            return t
        pad = (target_len - L, 0)
        return F.pad(t, pad, value=pad_value)
    
    @staticmethod
    def _right_pad_1d(t: torch.Tensor, target_len: int, pad_value):
        L = t.size(0)
        if L == target_len:
            return t
        pad = (0, target_len - L)  # (left, right) -> 右侧补零
        return F.pad(t, pad, value=pad_value)
        
    @staticmethod
    def collate_fn(batch: List[Dict], pad_token_id: int = None) -> Dict[str, torch.Tensor]:
        pad_id = 0 if pad_token_id is None else int(pad_token_id)

        # 1) 先各自最大
        max_a = max(x['audio_input_ids'].size(0) for x in batch)
        max_t = max(x['text_input_ids'].size(0)  for x in batch)
        # 2) 统一到同一个 L
        L = max(max_a, max_t)

        # helper：left padding function
        LP1 = LazySupervisedDataset._left_pad_1d
        LPB = LazySupervisedDataset._left_pad_bool

        # ---- tokens & masks left padding----
        audio_input_ids = torch.stack([LP1(x['audio_input_ids'], L, pad_id) for x in batch], dim=0)
        text_input_ids  = torch.stack([LP1(x['text_input_ids'],  L, pad_id) for x in batch], dim=0)
        is_continuous_mask = torch.stack([LPB(x['is_continuous_mask'], L, False) for x in batch], dim=0)

        audio_labels = torch.stack([LP1(x['labels'][0], L, pad_id) for x in batch], dim=0)
        text_labels  = torch.stack([LP1(x['labels'][1], L, pad_id) for x in batch], dim=0)
        audio_loss_mask = torch.stack([LPB(x['labels'][2], L, False) for x in batch], dim=0)
        text_loss_mask  = torch.stack([LPB(x['labels'][3], L, False) for x in batch], dim=0)

        # ---- waves right padding ----
        wav_lens = [x['waveform'].numel() for x in batch]
        max_w = max(wav_lens) if wav_lens else 0
        if max_w > 0:
            padded_waveforms = torch.stack([
                LazySupervisedDataset._right_pad_1d(x['waveform'], max_w, 0.0) for x in batch
            ], dim=0)
        else:
            padded_waveforms = torch.zeros(len(batch), 0)

        # ---- attention_mask 与位置编码将使用统一长度 L ----
        nonpad_audio = (audio_input_ids != pad_id)
        nonpad_text  = (text_input_ids  != pad_id)
        attention_mask = (nonpad_audio | nonpad_text)

        return dict(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            is_continuous_mask=is_continuous_mask,
            waveform=padded_waveforms,
            waveform_lengths=torch.tensor(wav_lens, dtype=torch.long),
            attention_mask=attention_mask,
            labels=(audio_labels, text_labels, audio_loss_mask, text_loss_mask),
        )