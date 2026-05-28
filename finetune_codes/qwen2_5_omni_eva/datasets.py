# datasets.py
# coding=utf-8
# Qwen2.5-Omni-EvA dataset processing

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
AUDIO_TOKEN = "<|AUDIO|>"
AUDIO_SPAN = "<|audio_bos|><|AUDIO|><|audio_eos|>"

MIN_AUDIO_SAMPLES = 1024  # 64ms @ 16kHz，低于此值 CED STFT 不可靠
MIN_AUDIO_DURATION = MIN_AUDIO_SAMPLES / 16000  # 换算为秒，与原始采样率无关


def _qwen2_5_omni_audio_output_len(mel_len: int) -> int:
    """Match Qwen2.5-Omni audio token count from valid mel-frame length."""
    if mel_len <= 0:
        return 0
    return int((((mel_len - 1) // 2 + 1 - 2) // 2) + 1)


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


class Qwen2_5OmniEvaDataset(Dataset):
    """
    Qwen2.5-Omni-EvA dataset.

    Returns per sample:
      input_ids            [L]
      attention_mask       [L]
      labels               [L]
      audio_mask           [L] bool    (audio token positions)
      input_features       [128, T_mel] mel features for audio_tower
      feature_attention_mask [T_mel]   valid mel frame mask
      audio_waveform       [T]         raw waveform for CED
      audio_waveform_length scalar     valid waveform samples before padding
    """

    def __init__(self, data_path, processor, max_length=8192, audio_token_index=151646):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.audio_token_index = audio_token_index
        self.data_dir = os.path.dirname(os.path.abspath(data_path))

        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
        self._resolve_relative_audio_paths(raw_data)
        logger.info(f"Loaded {len(raw_data)} examples")
        self.data = self._filter_bad_audio(raw_data)
        logger.info(f"After audio filtering: {len(self.data)} examples")

    def _resolve_relative_audio_paths(self, data):
        for item in data:
            conversations = item.get('conversations') or item.get('conversation', [])
            for turn in conversations:
                if turn.get('message_type') == 'audio':
                    content = turn.get('content')
                    if content and not os.path.isabs(content):
                        turn['content'] = os.path.normpath(os.path.join(self.data_dir, content))
                value = turn.get('value')
                if isinstance(value, str):
                    def repl(match):
                        path = match.group(1)
                        if os.path.isabs(path):
                            return match.group(0)
                        resolved = os.path.normpath(os.path.join(self.data_dir, path))
                        return f"<audio>{resolved}</audio>"
                    turn['value'] = re.sub(r'<audio>(.*?)</audio>', repl, value)
                content = turn.get('content')
                if isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'audio':
                            for key in ('audio_url', 'audio', 'path'):
                                path = part.get(key)
                                if path and not os.path.isabs(path):
                                    part[key] = os.path.normpath(os.path.join(self.data_dir, path))

    def _convert_fusiona_format(self, conversation):
        """FusionaData 格式 -> 标准 from/value 格式"""
        result = []
        current_role = None
        current_parts = []

        for msg in conversation:
            role = msg.get('role')
            msg_type = msg.get('message_type')
            content = msg.get('content', '')

            if role != current_role and current_parts:
                result.append({'from': current_role, 'value': '\n'.join(current_parts)})
                current_parts = []

            current_role = role

            if msg_type == 'text':
                current_parts.append(content)
            elif msg_type == 'audio':
                current_parts.append(f'<audio>{content}</audio>')

        if current_parts:
            result.append({'from': current_role, 'value': '\n'.join(current_parts)})

        return result

    def _extract_audio_path(self, text):
        """提取 <audio>path</audio> 中的路径，替换为官方 audio span 占位符。"""
        match = re.search(r'<audio>(.*?)</audio>', text)
        if match:
            replaced = text.replace(match.group(0), AUDIO_SPAN)
            return match.group(1), replaced
        return None, text

    def _get_audio_paths_from_item(self, item):
        """提取一个样本中所有音频路径，无音频返回空列表。"""
        paths = []
        conversations = item.get('conversations') or item.get('conversation', [])
        for turn in conversations:
            value = turn.get('value')
            if value is None and 'content' in turn:
                content = turn['content']
                if isinstance(content, list):
                    for ci in content:
                        if ci.get('type') == 'audio':
                            p = ci.get('audio_url') or ci.get('audio') or ci.get('path')
                            if p:
                                paths.append(p)
                    continue
                value = content if isinstance(content, str) else ''
            if value:
                m = re.search(r'<audio>(.*?)</audio>', value)
                if m:
                    paths.append(m.group(1))
        return paths

    def _filter_bad_audio(self, data):
        """过滤音频有问题的样本（文件不存在、损坏、过短）。无音频的纯文本样本保留。"""
        valid, n_bad = [], 0
        audio_ok_cache = {}
        for item in data:
            paths = self._get_audio_paths_from_item(item)
            paths_ok = True
            for p in paths:
                if p not in audio_ok_cache:
                    audio_ok_cache[p] = _check_audio_ok(p)
                if not audio_ok_cache[p]:
                    paths_ok = False
                    break
            if not paths or paths_ok:
                valid.append(item)
            else:
                n_bad += 1
        if n_bad:
            logger.warning(f"Filtered out {n_bad} samples with bad/missing/short audio")
        return valid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_item_impl(idx)

    def _get_item_impl(self, idx):
        item = self.data[idx]
        conversations = item.get('conversations') or item.get('conversation', [])
        if len(conversations) > 0 and 'message_type' in conversations[0]:
            conversations = self._convert_fusiona_format(conversations)

        audio_path = None
        audio_waveform = None
        turns = []
        for turn in conversations:
            role = turn.get('from') or turn.get('role')
            value = turn.get('value')
            if value is None and 'content' in turn:
                content = turn['content']
                if isinstance(content, list):
                    parts = []
                    for ci in content:
                        if ci['type'] == 'text':
                            parts.append(ci['text'])
                        elif ci['type'] == 'audio':
                            audio_val = ci.get("audio_url") or ci.get("audio") or ci.get("path")
                            if audio_val:
                                parts.append(f'<audio>{audio_val}</audio>')
                    value = ''.join(parts)
                else:
                    value = content
            path, processed = self._extract_audio_path(value)
            if path is not None:
                audio_path = path
            turns.append((role, processed))

        # load audio
        audio_array = None
        if audio_path is not None:
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found at __getitem__ time, skipping: {audio_path}")
                audio_path = None
                # strip <|AUDIO|> from text so audio_mask stays consistent
                turns = [(role, text.replace(AUDIO_TOKEN, '').strip()) for role, text in turns]
        if audio_path is not None:
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception:
                # fallback: try explicit mp3 format for files with wrong extension
                try:
                    waveform, sr = torchaudio.load(audio_path, format='mp3')
                except Exception as e:
                    logger.warning(f"Cannot load audio {audio_path}, skipping: {e}")
                    audio_path = None
                    turns = [(role, text.replace(AUDIO_TOKEN, '').strip()) for role, text in turns]
            if audio_path is not None:
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                audio_waveform = waveform.squeeze(0)
                audio_array = audio_waveform.numpy()

        # build chat text
        full_text = ""
        for role, text in turns:
            if role == 'user':
                full_text += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            elif role == 'assistant':
                full_text += f"{text}<|im_end|>\n"

        # processor: expand <|AUDIO|> tokens and extract mel features
        input_features = None
        feature_attention_mask = None
        if audio_array is not None and AUDIO_TOKEN in full_text:
            proc_out = self.processor(
                text=full_text, audio=[audio_array],
                sampling_rate=16000, return_tensors="pt", padding=True,
            )
            expanded_ids = proc_out['input_ids'][0].tolist()
            input_features = proc_out['input_features'][0]
            feature_attention_mask = proc_out['feature_attention_mask'][0]
        else:
            expanded_ids = self.processor.tokenizer.encode(full_text, add_special_tokens=False)

        # build labels: only compute loss on assistant tokens
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        assistant_ids = self.processor.tokenizer.encode('assistant', add_special_tokens=False)

        labels = [IGNORE_INDEX] * len(expanded_ids)
        i = 0
        while i < len(expanded_ids):
            if (expanded_ids[i] == im_start_id
                    and i + len(assistant_ids) < len(expanded_ids)
                    and expanded_ids[i+1:i+1+len(assistant_ids)] == assistant_ids):
                i += 1 + len(assistant_ids) + 1
                while i < len(expanded_ids):
                    labels[i] = expanded_ids[i]
                    if expanded_ids[i] == im_end_id:
                        i += 1
                        break
                    i += 1
            else:
                i += 1

        if len(expanded_ids) > self.max_length:
            expanded_ids = expanded_ids[:self.max_length]
            labels = labels[:self.max_length]
            if input_features is not None and feature_attention_mask is not None:
                mel_len = int(feature_attention_mask.sum().item())
                expected_audio_tokens = _qwen2_5_omni_audio_output_len(mel_len)
                kept_audio_tokens = sum(1 for token_id in expanded_ids if token_id == self.audio_token_index)
                if kept_audio_tokens != expected_audio_tokens:
                    logger.warning(
                        "Dropping audio features after max_length truncation caused partial audio tokens: "
                        f"kept={kept_audio_tokens}, expected={expected_audio_tokens}, max_length={self.max_length}"
                    )
                    audio_special_ids = {
                        self.audio_token_index,
                        self.processor.tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
                        self.processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
                    }
                    audio_special_ids.discard(None)
                    keep = [token_id not in audio_special_ids for token_id in expanded_ids]
                    expanded_ids = [token_id for token_id, keep_token in zip(expanded_ids, keep) if keep_token]
                    labels = [label for label, keep_token in zip(labels, keep) if keep_token]
                    input_features = None
                    feature_attention_mask = None
                    audio_waveform = None

        input_ids = torch.tensor(expanded_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        audio_mask = (input_ids == self.audio_token_index)
        if not audio_mask.any():
            audio_waveform = None

        waveform_length = 0
        if audio_waveform is not None:
            waveform_length = int(audio_waveform.shape[0])

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_t,
            'audio_mask': audio_mask,
            'audio_waveform': audio_waveform if audio_waveform is not None else torch.zeros(1),
            'audio_waveform_length': torch.tensor(waveform_length, dtype=torch.long),
        }
        if input_features is not None:
            result['input_features'] = input_features
            result['feature_attention_mask'] = feature_attention_mask
        return result


@dataclass
class Qwen2_5OmniEvaDataCollator:
    pad_token_id: int = 151643

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst['input_ids'] for inst in instances]
        labels = [inst['labels'] for inst in instances]
        attn_masks = [inst['attention_mask'] for inst in instances]
        audio_masks = [inst['audio_mask'] for inst in instances]

        max_len = max(len(x) for x in input_ids)
        batch_ids, batch_lbl, batch_attn, batch_amask = [], [], [], []
        for ids, lbl, attn, amask in zip(input_ids, labels, attn_masks, audio_masks):
            pad = max_len - len(ids)
            batch_ids.append(torch.cat([torch.full((pad,), self.pad_token_id, dtype=torch.long), ids]))
            batch_lbl.append(torch.cat([torch.full((pad,), IGNORE_INDEX, dtype=torch.long), lbl]))
            batch_attn.append(torch.cat([torch.zeros(pad, dtype=torch.long), attn]))
            batch_amask.append(torch.cat([torch.zeros(pad, dtype=torch.bool), amask]))

        result = {
            'input_ids': torch.stack(batch_ids),
            'attention_mask': torch.stack(batch_attn),
            'labels': torch.stack(batch_lbl),
            'audio_mask': torch.stack(batch_amask),
        }

        # input_features / feature_attention_mask: pad to max mel length
        has_features = any('input_features' in inst for inst in instances)
        if has_features:
            feats = []
            feat_masks = []
            max_mel = 0
            for inst in instances:
                if 'input_features' in inst:
                    max_mel = max(max_mel, inst['input_features'].shape[-1])
            for inst in instances:
                if 'input_features' in inst:
                    f = inst['input_features']
                    fm = inst['feature_attention_mask']
                    pad_mel = max_mel - f.shape[-1]
                    if pad_mel > 0:
                        f = torch.nn.functional.pad(f, (0, pad_mel))
                        fm = torch.nn.functional.pad(fm, (0, pad_mel))
                    feats.append(f)
                    feat_masks.append(fm)
                else:
                    feats.append(torch.zeros(128, max_mel))
                    feat_masks.append(torch.zeros(max_mel, dtype=torch.int32))
            result['input_features'] = torch.stack(feats)
            result['feature_attention_mask'] = torch.stack(feat_masks)

        # audio_waveform: right-pad to max length
        waveforms = [inst['audio_waveform'] for inst in instances]
        waveform_lengths = [inst.get('audio_waveform_length', torch.tensor(w.shape[0], dtype=torch.long)) for inst, w in zip(instances, waveforms)]
        max_wav = max(w.shape[0] for w in waveforms)
        batch_wav = []
        for w in waveforms:
            pad = max_wav - w.shape[0]
            batch_wav.append(torch.cat([w, torch.zeros(pad)]) if pad > 0 else w)
        result['audio_waveform'] = torch.stack(batch_wav)
        result['audio_waveform_lengths'] = torch.stack([
            l if torch.is_tensor(l) else torch.tensor(l, dtype=torch.long)
            for l in waveform_lengths
        ]).long()

        return result


def make_qwen2_5_omni_eva_data_module(
    data_path, processor, max_length=8192, audio_token_index=151646,
    eval_ratio=0.0, seed=42,
):
    dataset = Qwen2_5OmniEvaDataset(
        data_path=data_path, processor=processor,
        max_length=max_length, audio_token_index=audio_token_index,
    )

    if eval_ratio > 0:
        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(seed)
        eval_size = int(len(dataset) * eval_ratio)
        train_size = len(dataset) - eval_size
        train_dataset, eval_dataset = random_split(
            dataset, [train_size, eval_size], generator=generator)
    else:
        train_dataset = dataset
        eval_dataset = None

    return {'train_dataset': train_dataset, 'eval_dataset': eval_dataset}
