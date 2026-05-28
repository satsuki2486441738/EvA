# extract_semantic_codes.py
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # Compatibility imports.
from huggingface_hub import snapshot_download
import tqdm

import multiprocessing as mp
import torch
from typing import Tuple, List

# Optional faster JSON backend.
try:
    import orjson as fastjson
    def _loads(s): return fastjson.loads(s)
    def _dumps(o): return fastjson.dumps(o).decode("utf-8")
except Exception:
    _loads = json.loads
    def _dumps(o): return json.dumps(o, ensure_ascii=False)

from kimia_infer.api.prompt_manager import KimiAPromptManager


def _read_all_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


def _resume_state(output_file: str) -> Tuple[int, bool]:
    """
    Return (completed_line_count, whether_the_last_line_was_repaired).
    If the output file exists and the last JSONL line is corrupted, drop that
    partial line and return the repaired count.
    """
    if not os.path.exists(output_file):
        return 0, False

    # Read all lines for simplicity; this only runs during rare resume paths.
    lines = _read_all_lines(output_file)
    if not lines:
        return 0, False

    last = lines[-1].rstrip("\n")
    try:
        _ = _loads(last)
        return len(lines), False
    except Exception:
        # Drop the last partial or corrupted line.
        safe_lines = lines[:-1]
        with open(output_file, "w") as fw:
            for ln in safe_lines:
                fw.write(ln if ln.endswith("\n") else (ln + "\n"))
        return len(safe_lines), True


def _worker(rank, device_id, cache_path, kimia_token_offset, kimia_text_audiodelaytokens, lines, out_path):
    """
    Bind each worker to one GPU, process its shard, and write a part file.
    The output format matches the original input with audio_tokens added to
    messages where message_type == "audio".
    """
    torch.cuda.set_device(device_id)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    prompt_manager = KimiAPromptManager(
        model_path=cache_path,
        kimia_token_offset=kimia_token_offset,
        kimia_text_audiodelaytokens=kimia_text_audiodelaytokens
    )

    with open(out_path, "w", buffering=1024 * 1024) as f_out:
        pbar = tqdm.tqdm(lines, position=rank, leave=False, desc=f"GPU{device_id}")
        for line in pbar:
            data = _loads(line)
            conv = data.get("conversation", [])
            for msg in conv:
                if msg.get("message_type") == "audio":
                    audio_path = msg["content"]
                    try:
                        audio_tokens = prompt_manager._tokenize_audio(audio_path)
                        msg["audio_tokens"] = audio_tokens
                    except Exception as e:
                        msg["audio_tokens"] = None
                        msg["tokenize_error"] = str(e)
            f_out.write(_dumps(data) + "\n")


def main():
    # python -m kimi_audio_eva.extract_semantic_codes
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--audio_token_cache", type=str, default=None,
                        help="Directory for audio token cache (sets KIMIA_AUDIO_TOKEN_CACHE).")
    parser.add_argument("--gpus", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0,1').")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.audio_token_cache is not None:
        os.environ["KIMIA_AUDIO_TOKEN_CACHE"] = args.audio_token_cache

    # Resolve the model path.
    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)

    # Read the required config fields.
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    kimia_token_offset = getattr(model_config, "kimia_token_offset", 0)
    kimia_text_audiodelaytokens = getattr(model_config, "kimia_mimo_audiodelaytokens", 0)

    # Read input lines.
    with open(args.input_file, "r") as f:
        lines = f.readlines()
    total = len(lines)

    # Resume support: inspect existing output and continue from the next line.
    done_n, fixed = _resume_state(args.output_file)
    if done_n > 0:
        tqdm.tqdm.write(f"[resume] detected {done_n} completed line(s){' (fixed last line)' if fixed else ''}.")
    if done_n >= total:
        tqdm.tqdm.write("[resume] output already complete. nothing to do.")
        return

    # Lines that still need processing.
    remaining = lines[done_n:]

    num_gpus = torch.cuda.device_count()

    # Single-GPU path: append sequentially.
    if num_gpus <= 1:
        prompt_manager = KimiAPromptManager(
            model_path=cache_path,
            kimia_token_offset=kimia_token_offset,
            kimia_text_audiodelaytokens=kimia_text_audiodelaytokens
        )
        # Append mode is required for resume support.
        with open(args.output_file, "a", buffering=1024 * 1024) as f_out:
            for line in tqdm.tqdm(remaining, desc="GPU0"):
                data = _loads(line)
                for msg in data.get("conversation", []):
                    if msg.get("message_type") == "audio":
                        audio_path = msg["content"]
                        try:
                            audio_tokens = prompt_manager._tokenize_audio(audio_path)
                            msg["audio_tokens"] = audio_tokens
                        except Exception as e:
                            msg["audio_tokens"] = None
                            msg["tokenize_error"] = str(e)
                f_out.write(_dumps(data) + "\n")
        return

    # Multi-GPU path: shard remaining lines, then append part files in order.
    parts = []
    rem_total = len(remaining)
    for r in range(num_gpus):
        s = r * rem_total // num_gpus
        e = (r + 1) * rem_total // num_gpus
        parts.append((s, e))

    mp.set_start_method("spawn", force=True)

    # Remove stale part files left by interrupted runs.
    for rank in range(num_gpus):
        pf = f"{args.output_file}.part{rank:02d}"
        if os.path.exists(pf):
            try:
                os.remove(pf)
            except:
                pass

    procs, part_files = [], []
    for rank, (s, e) in enumerate(parts):
        out_part = f"{args.output_file}.part{rank:02d}"
        part_files.append(out_part)
        p = mp.Process(
            target=_worker,
            args=(
                rank,               # rank
                rank,               # device_id: 0..N-1 after CUDA_VISIBLE_DEVICES remapping
                cache_path,
                kimia_token_offset,
                kimia_text_audiodelaytokens,
                remaining[s:e],
                out_part,
            ),
            daemon=False
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Append part files to the main output in rank order.
    with open(args.output_file, "a", buffering=1024 * 1024) as f_out:
        for pf in part_files:
            with open(pf, "r") as fin:
                for line in fin:
                    f_out.write(line)

    # Clean up part files.
    for pf in part_files:
        try:
            os.remove(pf)
        except:
            pass


if __name__ == "__main__":
    main()
