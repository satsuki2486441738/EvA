import json
import os
import argparse
import logging
from typing import Set, Any, Tuple

logger = logging.getLogger(__name__)

def init_dist_if_needed() -> Tuple[int, int, int]:
    """
    若以 torchrun 启动，初始化进程组，并返回 (rank, local_rank, world_size)。
    否则返回 (0, 0, 1)。
    """
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        logger.info(f"[dist] initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return rank, local_rank, world_size
    return 0, 0, 1

def dist_barrier_if_needed():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """保存数据到 JSONL 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_done_ids(output_file: str) -> Set[Any]:
    """加载已完成的样本ID（用于断点续跑）"""
    done = set()
    if not os.path.exists(output_file):
        return done
    
    try:
        data = load_jsonl(output_file)
        for item in data:
            if isinstance(item, dict) and "sample_id" in item:
                done.add(item["sample_id"])
    except Exception:
        pass
    return done

def merge_parts(base_output: str, world_size: int):
    """
    rank0 合并 part 文件到 JSONL 格式：
    base_output: /path/to/output.jsonl
    期望存在: /path/to/output.jsonl.part{0..world_size-1}
    """
    base_dir = os.path.dirname(base_output)
    base_name = os.path.basename(base_output)
    part_paths = []
    for r in range(world_size):
        part_paths.append(os.path.join(base_dir, f"{base_name}.part{r}"))
    
    merged_data = []
    for p in part_paths:
        if not os.path.exists(p):
            logger.warning(f"[merge] missing part: {p}")
            continue
        try:
            part_data = load_jsonl(p)
            merged_data.extend(part_data)
        except Exception as e:
            logger.warning(f"[merge] failed to load part {p}: {e}")
            continue
    
    save_jsonl(merged_data, base_output)
    logger.success(f"[merge] merged into: {base_output}")

def process_conversation(conversation):
    """处理对话，移除 assistant 的回复，保留用户输入，将assistant改为reference"""
    chats = []
    reference_content = None
    
    for message in conversation:
        if message["role"] == "user":
            chats.append({
                "role": message["role"],
                "message_type": message["message_type"],
                "content": message["content"]
            })
        elif message["role"] == "assistant":
            # 将原有的assistant回复保存为reference
            reference_content = message.get("content", "")
    
    return chats, reference_content


def resolve_relative_audio_paths(data, input_file):
    data_dir = os.path.dirname(os.path.abspath(input_file))
    for sample in data:
        for message in sample.get("conversation", []):
            if message.get("message_type") == "audio":
                content = message.get("content")
                if content and not os.path.isabs(content):
                    message["content"] = os.path.normpath(os.path.join(data_dir, content))
    return data

def run_dataset_inference(input_file, output_file, model_path, max_new_tokens=128, text_temperature=0.0,
                         resume=False, gather_on_rank0=False, show_every=20):
    """
    运行数据集推理（支持多GPU torchrun分片）
    """
    import torch
    from tqdm import tqdm
    from kimia_infer.api.kimia import KimiAudio

    # --- 分布式初始化（若以 torchrun 启动） ---
    rank, local_rank, world_size = init_dist_if_needed()
    use_dist = (world_size > 1)

    # --- 准备输出 part 文件名 ---
    base_out = output_file
    if use_dist:
        part_out = f"{base_out}.part{rank}"
    else:
        part_out = base_out  # 单进程单卡，直接写最终文件

    logger.info(f"[rank={rank}] --- Starting Dataset Inference ---")
    
    # 加载输入数据
    logger.info(f"[rank={rank}] Loading input data from: {input_file}")
    input_data = load_jsonl(input_file)
    input_data = resolve_relative_audio_paths(input_data, input_file)
    logger.info(f"[rank={rank}] Loaded {len(input_data)} samples")
    
    # 初始化模型
    logger.info(f"[rank={rank}] Initializing KimiAudio API from '{model_path}'...")
    try:
        kimia_api = KimiAudio(model_path=model_path, load_detokenizer=False)
        logger.info(f"[rank={rank}] KimiAudio API initialized successfully")
    except Exception as e:
        logger.error(f"[rank={rank}] Failed to initialize model: {e}")
        return

    # --- 断点续跑 (针对当前 part 文件) ---
    done_ids: Set[Any] = set()
    if resume and os.path.exists(part_out):
        done_ids = load_done_ids(part_out)
        logger.info(f"[rank={rank}] Resume: loaded {len(done_ids)} done ids from {part_out}")
    
    # 处理每个样本
    results = []
    total, selected, skipped, written = 0, 0, 0, 0
    
    for idx, sample in enumerate(tqdm(input_data, desc=f"Processing@rank{rank}")):
        total += 1

        # 分片选择逻辑
        if use_dist:
            # torchrun: 按 world_size/rank 分片
            if (idx % world_size) != rank:
                continue

        selected += 1

        try:
            # 提取任务类型和对话
            task_type = sample.get("task_type", "understanding")
            conversation = sample["conversation"]
            sample_id = sample.get("sample_id", idx)  # 使用样本ID或索引作为唯一标识
            
            # 断点续跑检查
            if resume and sample_id in done_ids:
                skipped += 1
                continue
            
            # 处理对话，只保留用户输入，提取reference
            chats, reference_content = process_conversation(conversation)
            
            # 检查音频文件是否存在
            audio_path = None
            for chat in chats:
                if chat["message_type"] == "audio":
                    audio_path = chat["content"]
                    break
            
            if audio_path and not os.path.exists(audio_path):
                logger.warning(f"[rank={rank}] Audio file not found: {audio_path}, skipping sample {sample_id}")
                continue
            
            # 运行推理
            with torch.inference_mode():
                generated_wav, generated_text = kimia_api.generate(
                    chats=chats,
                    output_type="text",
                    text_temperature=text_temperature,
                    max_new_tokens=max_new_tokens,
                    text_repetition_penalty=1.05
                )
                print(f"generated_text:{generated_text}")
            # 构造输出格式
            output_conversation = chats.copy()
            
            # 添加reference（原有的assistant回复）
            if reference_content:
                output_conversation.append({
                    "role": "reference",
                    "message_type": "text", 
                    "content": reference_content
                })
            
            # 添加模型的真实输出作为assistant回复
            output_conversation.append({
                "role": "assistant",
                "message_type": "text", 
                "content": generated_text
            })
            
            result = {
                "sample_id": sample_id,
                "task_type": task_type,
                "conversation": output_conversation
            }
            
            results.append(result)
            written += 1
            
            if written % show_every == 0:
                logger.info(f"[rank={rank}] Processed {written} samples | latest: {sample_id}")
                
        except Exception as e:
            import traceback
            logger.error(f"[rank={rank}] Failed to process sample {idx} (sample_id={sample.get('sample_id', idx)}): {e}")
            traceback.print_exc()
            continue
    
    # 保存结果到 part 文件
    logger.info(f"[rank={rank}] Saving results to: {part_out}")
    save_jsonl(results, part_out)
    
    logger.info(f"[rank={rank}] DONE total={total}, selected={selected}, skipped={skipped}, written={written}")

    # --- 可选：rank0 合并 ---
    if gather_on_rank0 and use_dist:
        dist_barrier_if_needed()
        if rank == 0:
            merge_parts(output_file, world_size)
        dist_barrier_if_needed()

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Dataset inference with KimiAudio (Multi-GPU)")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--model_path", type=str,
                       required=True,
                       help="Model path")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--text_temperature", type=float, default=0.0,
                       help="Temperature for text generation")
    
    # 其他参数
    parser.add_argument("--resume", action="store_true", 
                       help="Resume on each part file")
    parser.add_argument("--gather_on_rank0", action="store_true", 
                       help="After torchrun, rank0 merges all parts")
    parser.add_argument("--show_every", type=int, default=20, 
                       help="Log every N samples")
    
    args = parser.parse_args()
    
    # 运行推理
    run_dataset_inference(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        text_temperature=args.text_temperature,
        resume=args.resume,
        gather_on_rank0=args.gather_on_rank0,
        show_every=args.show_every
    )

if __name__ == "__main__":
    main()
