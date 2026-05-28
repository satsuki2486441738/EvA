#!/usr/bin/env python3
# coding=utf-8

import argparse
import os
import sys

# third_party contains kimia_infer (CedEncoder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party'))


def load_audio(path, sampling_rate=16000):
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != sampling_rate:
        waveform = torchaudio.transforms.Resample(sr, sampling_rate)(waveform)
    return waveform.squeeze(0)


def apply_qwen_omni_chat_template(processor, conversation):
    chat_template = getattr(processor, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(processor.tokenizer, "chat_template", None)
    if chat_template is None:
        raise ValueError("Qwen2.5-Omni processor/tokenizer has no chat template")
    return processor.apply_chat_template(
        conversation,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=False,
    )


def move_inputs_to_device(inputs, device, dtype=None):
    import torch

    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif dtype is not None and torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def load_model(model_path, adapter_path=None, ced_model_path=None, device="cuda"):
    import torch
    from qwen2_5_omni_eva.configuration_qwen2_5_omni_eva import Qwen2_5OmniEvaConfig
    from qwen2_5_omni_eva.modeling_qwen2_5_omni_eva import Qwen2_5OmniEvaForConditionalGeneration

    config = Qwen2_5OmniEvaConfig.from_pretrained(model_path)
    model = Qwen2_5OmniEvaForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if adapter_path:
        from qwen2_5_omni_eva.lora_utils import load_adapter_for_inference
        model = load_adapter_for_inference(model, adapter_path, merge=False)

    ced_path = ced_model_path or os.path.join(model_path, "ced-base")
    if not os.path.exists(ced_path) and adapter_path:
        ced_path = os.path.join(adapter_path, "ced-base")
    if os.path.exists(ced_path):
        from kimia_infer.models.tokenizer.ced_base.modeling_ced import CedEncoder
        model.set_ced_model(CedEncoder(ced_path))

    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = config.eos_token_id
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = config.pad_token_id
    model.eval()
    return model


def build_inputs(processor, prompt, audio_path, device):
    import torch

    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]
    text = apply_qwen_omni_chat_template(processor, conversation)
    waveform = load_audio(audio_path, sampling_rate=16000)
    audio_array = waveform.cpu().numpy()
    proc_out = processor(
        text=text,
        audio=[audio_array],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_ids = proc_out["input_ids"]
    audio_token_id = getattr(processor.tokenizer, "audio_token_id", 151646)
    if audio_token_id is None:
        audio_token_id = 151646
    inputs = {
        "input_ids": input_ids,
        "attention_mask": proc_out["attention_mask"],
        "input_features": proc_out["input_features"],
        "feature_attention_mask": proc_out["feature_attention_mask"],
        "audio_waveform": waveform.unsqueeze(0),
        "audio_waveform_lengths": torch.tensor([waveform.shape[0]], dtype=torch.long),
        "audio_mask": input_ids == audio_token_id,
    }
    return move_inputs_to_device(inputs, device)


def main():
    parser = argparse.ArgumentParser(description="Single-sample inference with Qwen2.5-Omni-EvA")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--ced_model_path", default=None)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--prompt", default="Narrate the sound scene.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import Qwen2_5OmniProcessor

    processor_path = args.adapter_path or args.model_path
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = load_model(args.model_path, args.adapter_path, args.ced_model_path, args.device)
    inputs = build_inputs(processor, args.prompt, args.audio_path, args.device)
    inputs = move_inputs_to_device(inputs, args.device, dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
        )
    output_ids = output_ids[0] if isinstance(output_ids, tuple) else output_ids
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:] if output_ids.shape[1] > prompt_len else output_ids
    print(processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
