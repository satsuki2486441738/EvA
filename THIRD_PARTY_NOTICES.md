# Third-Party Notices

This file summarizes third-party code that is included in this repository.
It is informational and does not replace the license text in the referenced
files or upstream projects.

## Included Third-Party Code

The `src/third_party/kimia_infer/` tree contains compatibility and inference
code used by the Kimi-Audio-EvA path, including tokenizer, CED, Whisper,
detokenizer, GLM-4-Voice and CosyVoice-related components.

The following license evidence is present in the repository:

| Path / component | License evidence in this repo |
|---|---|
| `src/third_party/kimia_infer/models/tokenizer/glm4/` | Includes `LICENSE`, Apache License 2.0. |
| `src/third_party/kimia_infer/models/tokenizer/ced_base/` | File headers state Apache License 2.0 and copyright Xiaomi / Hugging Face. |
| `src/third_party/kimia_infer/models/tokenizer/whisper_Lv3/` | File headers state Apache License 2.0 and copyright OpenAI / Hugging Face. |
| `src/third_party/kimia_infer/models/tokenizer/glm4/cosyvoice/` | Many file headers state Apache License 2.0 and upstream authors from Alibaba, Mobvoi, Johns Hopkins, ESPnet-related contributors, or Hugging Face. |
| `src/third_party/kimia_infer/models/detokenizer/vocoder/alias_free_activation/` | File headers mention Apache License 2.0 for alias-free-torch-derived code and MIT-derived filtering utilities. |
| `src/third_party/kimia_infer/models/detokenizer/vocoder/bigvgan.py` and CUDA activation files | File headers include NVIDIA copyright notices. |
| Selected files under `src/third_party/kimia_infer/models/tokenizer/glm4/cosyvoice/flow/stable/` | File headers mention MIT-licensed adaptations from x-transformers, audio-diffusion-pytorch, AudioCraft, or NVIDIA BigVGAN components. Short SPDX notices are included under `flow/stable/LICENSES/`. |

## Model Weights

This repository does not include external backbone, tokenizer, CED, Whisper,
Kimi-Audio, Qwen2.5-Omni, GLM-4-Voice, or EvA checkpoint weights. Those weights
must be obtained from their own distribution channels and used under their
respective model licenses or terms.
