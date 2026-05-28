import json
import os
import argparse
import logging
from typing import Set, Any, Tuple

logger = logging.getLogger(__name__)

def init_dist_if_needed() -> Tuple[int, int, int]:
    """
    Initialize torch.distributed when launched with torchrun and return
    (rank, local_rank, world_size). Otherwise return (0, 0, 1).
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
    """Load a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_done_ids(output_file: str) -> Set[Any]:
    """Load completed sample IDs for resume support."""
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
    Rank 0 merges part files into a JSONL file.
    base_output: /path/to/output.jsonl
    Expected parts: /path/to/output.jsonl.part{0..world_size-1}
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
    """Keep user messages and move the assistant answer into a reference field."""
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
            # Save the original assistant reply as reference.
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
    Run dataset inference with optional multi-GPU torchrun sharding.
    """
    import torch
    from tqdm import tqdm
    from kimia_infer.api.kimia import KimiAudio

    # Distributed initialization when launched with torchrun.
    rank, local_rank, world_size = init_dist_if_needed()
    use_dist = (world_size > 1)

    # Prepare output part filename.
    base_out = output_file
    if use_dist:
        part_out = f"{base_out}.part{rank}"
    else:
        part_out = base_out  # Single process/GPU: write directly to final output.

    logger.info(f"[rank={rank}] --- Starting Dataset Inference ---")
    
    # Load input data.
    logger.info(f"[rank={rank}] Loading input data from: {input_file}")
    input_data = load_jsonl(input_file)
    input_data = resolve_relative_audio_paths(input_data, input_file)
    logger.info(f"[rank={rank}] Loaded {len(input_data)} samples")
    
    # Initialize model.
    logger.info(f"[rank={rank}] Initializing KimiAudio API from '{model_path}'...")
    try:
        kimia_api = KimiAudio(model_path=model_path, load_detokenizer=False)
        logger.info(f"[rank={rank}] KimiAudio API initialized successfully")
    except Exception as e:
        logger.error(f"[rank={rank}] Failed to initialize model: {e}")
        return

    # Resume support for the current part file.
    done_ids: Set[Any] = set()
    if resume and os.path.exists(part_out):
        done_ids = load_done_ids(part_out)
        logger.info(f"[rank={rank}] Resume: loaded {len(done_ids)} done ids from {part_out}")
    
    # Process each sample.
    results = []
    total, selected, skipped, written = 0, 0, 0, 0
    
    for idx, sample in enumerate(tqdm(input_data, desc=f"Processing@rank{rank}")):
        total += 1

        # Shard selection.
        if use_dist:
            # torchrun: shard by world_size/rank.
            if (idx % world_size) != rank:
                continue

        selected += 1

        try:
            # Extract task type and conversation.
            task_type = sample.get("task_type", "understanding")
            conversation = sample["conversation"]
            sample_id = sample.get("sample_id", idx)  # Use sample_id or index as the unique identifier.
            
            # Resume check.
            if resume and sample_id in done_ids:
                skipped += 1
                continue
            
            # Keep user inputs and extract the reference answer.
            chats, reference_content = process_conversation(conversation)
            
            # Check whether the audio file exists.
            audio_path = None
            for chat in chats:
                if chat["message_type"] == "audio":
                    audio_path = chat["content"]
                    break
            
            if audio_path and not os.path.exists(audio_path):
                logger.warning(f"[rank={rank}] Audio file not found: {audio_path}, skipping sample {sample_id}")
                continue
            
            # Run inference.
            with torch.inference_mode():
                generated_wav, generated_text = kimia_api.generate(
                    chats=chats,
                    output_type="text",
                    text_temperature=text_temperature,
                    max_new_tokens=max_new_tokens,
                    text_repetition_penalty=1.05
                )
                print(f"generated_text:{generated_text}")
            # Build output format.
            output_conversation = chats.copy()
            
            # Add reference, which is the original assistant reply.
            if reference_content:
                output_conversation.append({
                    "role": "reference",
                    "message_type": "text", 
                    "content": reference_content
                })
            
            # Add the model output as the assistant reply.
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
    
    # Save results to the part file.
    logger.info(f"[rank={rank}] Saving results to: {part_out}")
    save_jsonl(results, part_out)
    
    logger.info(f"[rank={rank}] DONE total={total}, selected={selected}, skipped={skipped}, written={written}")

    # Optional rank-0 merge.
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
    
    # Additional options.
    parser.add_argument("--resume", action="store_true", 
                       help="Resume on each part file")
    parser.add_argument("--gather_on_rank0", action="store_true", 
                       help="After torchrun, rank0 merges all parts")
    parser.add_argument("--show_every", type=int, default=20, 
                       help="Log every N samples")
    
    args = parser.parse_args()
    
    # Run inference.
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
