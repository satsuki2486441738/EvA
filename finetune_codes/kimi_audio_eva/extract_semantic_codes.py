# extract_semantic_codes.py
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # 兼容环境
from huggingface_hub import snapshot_download
import tqdm

import multiprocessing as mp
import torch
from typing import Tuple, List

# 更快的 json（可选）
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
    返回 (已完成行数, 是否修复过最后一行)。
    逻辑：
      - 文件不存在 -> (0, False)
      - 存在 -> 统计行数 N；若最后一行 JSON 解析失败，则将其丢弃（覆盖写回），返回 (N-1, True)
    """
    if not os.path.exists(output_file):
        return 0, False

    # 尽量只读最后一行；为简洁与稳健，这里读取全部行（发生在崩溃场景下且很少）
    lines = _read_all_lines(output_file)
    if not lines:
        return 0, False

    last = lines[-1].rstrip("\n")
    try:
        _ = _loads(last)
        return len(lines), False
    except Exception:
        # 丢弃最后一行（半截/损坏）
        safe_lines = lines[:-1]
        with open(output_file, "w") as fw:
            for ln in safe_lines:
                fw.write(ln if ln.endswith("\n") else (ln + "\n"))
        return len(safe_lines), True


def _worker(rank, device_id, cache_path, kimia_token_offset, kimia_text_audiodelaytokens, lines, out_path):
    """
    每个进程绑定一张 GPU，处理自己的分片，输出到分片文件。
    输出格式与原脚本一致：对 message_type=="audio" 的消息添加 audio_tokens 字段。
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

    # 准备模型路径
    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)

    # 读取必要的 config 字段
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    kimia_token_offset = getattr(model_config, "kimia_token_offset", 0)
    kimia_text_audiodelaytokens = getattr(model_config, "kimia_mimo_audiodelaytokens", 0)

    # 读取输入
    with open(args.input_file, "r") as f:
        lines = f.readlines()
    total = len(lines)

    # —— 断点续传：检查已有输出，决定从第几行续 —— #
    done_n, fixed = _resume_state(args.output_file)
    if done_n > 0:
        tqdm.tqdm.write(f"[resume] detected {done_n} completed line(s){' (fixed last line)' if fixed else ''}.")
    if done_n >= total:
        tqdm.tqdm.write("[resume] output already complete. nothing to do.")
        return

    # 待处理的剩余行
    remaining = lines[done_n:]

    num_gpus = torch.cuda.device_count()

    # 单卡：顺序追加
    if num_gpus <= 1:
        prompt_manager = KimiAPromptManager(
            model_path=cache_path,
            kimia_token_offset=kimia_token_offset,
            kimia_text_audiodelaytokens=kimia_text_audiodelaytokens
        )
        # 以“追加”方式写入（断点续传关键点）
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

    # 多卡：把“剩余行”分片并行，处理完按序追加到 output_file
    parts = []
    rem_total = len(remaining)
    for r in range(num_gpus):
        s = r * rem_total // num_gpus
        e = (r + 1) * rem_total // num_gpus
        parts.append((s, e))

    mp.set_start_method("spawn", force=True)

    # 清理可能存在的旧分片文件（避免上次崩溃遗留）
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
                rank,               # device_id: 0..N-1 对应 CUDA_VISIBLE_DEVICES
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

    # 以“追加”方式合并到主输出（断点续传关键点）
    with open(args.output_file, "a", buffering=1024 * 1024) as f_out:
        for pf in part_files:
            with open(pf, "r") as fin:
                for line in fin:
                    f_out.write(line)

    # 清理分片
    for pf in part_files:
        try:
            os.remove(pf)
        except:
            pass


if __name__ == "__main__":
    main()
