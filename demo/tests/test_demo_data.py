import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo" / "data" / "audio_understanding"


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_demo_jsonl_audio_paths_exist():
    for name in ["data.jsonl", "data_with_semantic_codes.jsonl"]:
        rows = _load_jsonl(DEMO_DIR / name)
        assert rows, f"{name} is empty"
        for row in rows:
            for message in row["conversation"]:
                if message.get("message_type") == "audio":
                    assert (DEMO_DIR / message["content"]).is_file()


def test_kimi_demo_data_has_semantic_audio_tokens():
    rows = _load_jsonl(DEMO_DIR / "data_with_semantic_codes.jsonl")
    for row in rows:
        audio_messages = [
            message
            for message in row["conversation"]
            if message.get("message_type") == "audio"
        ]
        assert audio_messages
        for message in audio_messages:
            assert isinstance(message.get("audio_tokens"), list)
            assert message["audio_tokens"]
            assert all(isinstance(token, int) for token in message["audio_tokens"])
