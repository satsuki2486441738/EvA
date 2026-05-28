import importlib

import pytest


def test_lightweight_first_party_imports():
    pytest.importorskip("transformers")

    for module_name in [
        "eva_processor",
        "kimi_audio_eva.configuration_moonshot_kimia",
        "qwen2_5_omni_eva.configuration_qwen2_5_omni_eva",
    ]:
        importlib.import_module(module_name)
