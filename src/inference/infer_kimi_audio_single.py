# infer_kimi_audio_single.py
import os
import argparse


def run_inference(model_path: str, audio_paths: list, prompt: str, max_new_tokens: int):
    from loguru import logger
    from kimia_infer.api.kimia import KimiAudio

    logger.info(f"Initializing KimiAudio API from '{model_path}'...")
    kimia_api = KimiAudio(model_path=model_path, load_detokenizer=False)
    logger.success("KimiAudio API initialized successfully.")

    for idx, audio_path in enumerate(audio_paths, 1):
        logger.info(f"\n--- Processing Audio {idx}/{len(audio_paths)}: {audio_path} ---")
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}, skipping...")
            continue

        chats = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]

        try:
            _, generated_text = kimia_api.generate(
                chats=chats,
                output_type="text",
                text_temperature=0.0,
                max_new_tokens=max_new_tokens,
            )
            print("\n" + "=" * 70)
            print(f"AUDIO {idx}: {os.path.basename(audio_path)}")
            print("-" * 70)
            print(f"Generated Text:\n---\n{generated_text}\n---")
            print("=" * 70)
        except Exception as e:
            logger.error(f"Failed to process audio {idx} ({audio_path}): {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sample inference with KimiAudio")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the KimiAudio model directory.")
    parser.add_argument("--audio", type=str, nargs="+", required=True,
                        help="One or more audio file paths.")
    parser.add_argument("--prompt", type=str,
                        default="Describe the audio content in detail.\n",
                        help="Text prompt sent before the audio.")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--gpus", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0,1').")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    run_inference(args.model_path, args.audio, args.prompt, args.max_new_tokens)
