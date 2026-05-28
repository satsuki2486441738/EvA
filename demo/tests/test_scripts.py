import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_training_scripts_show_help():
    scripts = [
        ROOT / "src" / "scripts" / "train_kimi_audio_eva.sh",
        ROOT / "src" / "scripts" / "train_qwen2_5_omni_eva.sh",
    ]
    for script in scripts:
        result = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "Usage:" in result.stdout
        assert "--model" in result.stdout
        assert "--data" in result.stdout
        assert "--output" in result.stdout
