# Third-party notices

qwen3-tts-lowvram ships **only its own code** (Apache-2.0, see [LICENSE](LICENSE)). Nothing below is
vendored or redistributed in this repository: every package is installed by you from PyPI, and the model
weights are downloaded by you from Hugging Face on first run. Each keeps its own license, which you
accept when you install or download it.

## Models

| Component | Source | License |
|---|---|---|
| Qwen3-TTS-12Hz-0.6B-Base (default) and other Qwen3-TTS checkpoints | [huggingface.co/Qwen](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Apache-2.0 |

The weights are not part of this repository. Check the model card of the exact checkpoint you use;
the license above was verified for the 0.6B and 1.7B Base checkpoints on 2026-10-16.

## Runtime dependencies

| Package | Used for | Upstream | License |
|---|---|---|---|
| faster-qwen3-tts 0.4.0 | CUDA-graph Qwen3-TTS runtime that `optimize()` patches | [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) | MIT, © 2026 Andres Marafioti |
| qwen-tts / qwen-tts-hf | Qwen3-TTS model code (pulled in by faster-qwen3-tts) | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache-2.0 |
| PyTorch | tensors, CUDA | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause |
| NumPy | audio buffers | [numpy/numpy](https://github.com/numpy/numpy) | BSD-3-Clause |

## Optional dependencies

| Package | Used by | Upstream | License |
|---|---|---|---|
| FastAPI | `examples/server.py` | [fastapi/fastapi](https://github.com/fastapi/fastapi) | MIT |
| Uvicorn | `examples/server.py` | [Kludex/uvicorn](https://github.com/Kludex/uvicorn) | BSD-3-Clause |
| Pydantic | `examples/server.py` | [pydantic/pydantic](https://github.com/pydantic/pydantic) | MIT |
| python-sounddevice | `examples/server.py` (local playback) | [spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice) | MIT |
| python-soundfile | `bench/` | [bastibe/python-soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause |
| torchao | `bench/ --quant` (experimental) | [pytorch/ao](https://github.com/pytorch/ao) | BSD-3-Clause |

## Voice cloning

The repository includes no voice recordings. You supply your own reference audio, and you are
responsible for having the right to use that voice. Do not clone someone's voice without their
consent, and do not use generated speech to impersonate or deceive.
