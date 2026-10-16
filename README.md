# qwen3-tts-lowvram

Cut [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) VRAM roughly in half without changing the voice,
the weights' precision or the output: one call on a loaded
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) model.

```python
from faster_qwen3_tts import FasterQwen3TTS
from qwen3_tts_lowvram import optimize

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", max_seq_len=512)
optimize(model)  # before the first generation

for chunk, sr, _ in model.generate_voice_clone_streaming(
    text="Hello there.", language="English",
    ref_audio="my_voice.wav", ref_text="Exact transcript of my_voice.wav.",
):
    ...
```

## Results

Qwen3-TTS-12Hz-0.6B-Base, bf16, voice clone from a 17 s reference, streaming with `chunk_size=8`,
RTX 4060 8 GB, Windows 11, torch 2.12 + CUDA 13.0. Same seed for every row, so the audio is
directly comparable.

| config | idle VRAM | peak VRAM | RTF | first chunk |
|---|---|---|---|---|
| stock faster-qwen3-tts | 3.16 GB | 3.59 GB | 0.66 | 604 ms |
| embedding offload + repack | 1.83 GB | 3.25 GB | 0.67 | 627 ms |
| + decode context 25 | 1.83 GB | 2.01 GB | 0.53 | 425 ms |
| **`optimize()` (+ 2.5 GB cap)** | **1.84 GB** | **2.02 GB** | **0.52** | **428 ms** |

VRAM is the whole-GPU `memory.used` delta from before the model loaded, so it includes the CUDA
context. RTF is generation time divided by audio length (lower is faster). The bf16 rows produce
identical line lengths, and no row shows clicks at chunk joins. Reproduce with `bench/` (below).

## What it does

1. **Text embedding to CPU (−594 MiB).** The talker's text embedding is a 151936 × 2048 bf16
   table. It is only used to build the prompt and never runs inside the CUDA-graph decode loop,
   so rows are looked up in RAM and only the result vectors go to the GPU.
2. **Repack the weights (−~0.7 GB of reserved memory).** Freeing one tensor leaves a hole inside an
   allocator segment that other weights still pin, so `torch.cuda.empty_cache()` can't return it.
   Every GPU weight, including the codec (`speech_tokenizer`, which isn't a registered submodule),
   is moved to CPU once and re-uploaded into tightly packed segments.
3. **Trim the codec decode context (−~400 MB peak, ~180 ms faster first chunk).** In voice-clone
   mode each line starts by running the *whole* reference (~210 frames for 17 s) through the codec
   decoder. Only the last 25 frames (2 s) are kept as acoustic context now. The talker prompt is
   still built from the full cached reference, so the voice doesn't change.
4. **Per-process VRAM cap (2.5 GB).** When the caching allocator would grow past it, it frees cached
   blocks and retries instead. This stands in for `expandable_segments`, which Windows doesn't have.

Each step is also available on its own: `offload_text_embedding`, `repack_weights`,
`trim_decode_context`, `set_vram_cap`. For a long-running server, call `torch.cuda.empty_cache()`
between lines so idle memory goes back to ~1.8 GB.

```python
optimize(model, decode_context=25, vram_cap_gb=2.5)   # defaults
optimize(model, vram_cap_gb=None)                     # no cap (e.g. the 1.7B model)
```

## Example server

`examples/server.py` is a FastAPI voice-clone server built around `optimize()`: queued local playback,
stop with fade-out, WAV and streamed PCM.

```bash
pip install -e ".[server]"
python examples/server.py --ref-audio my_voice.wav --ref-text "Exact transcript." --port 2027
curl -X POST localhost:2027/synthesize -H "Content-Type: application/json" -d '{"text":"Hi!"}' -o out.wav
```

| Endpoint | |
|---|---|
| `GET /ping` | health check |
| `POST /play` `{"text"}` | queue a line for playback on the local audio device (`--output-device`) |
| `POST /stop` | clear the queue, fade out |
| `GET /is_playing` | `{"is_playing": bool}` |
| `POST /synthesize` `{"text"}` | WAV, 24 kHz mono 16-bit |
| `POST /synthesize_stream` `{"text"}` | raw s16le PCM, 24 kHz mono, chunk by chunk |

It listens on 127.0.0.1 by default. Use `--host 0.0.0.0` only on a trusted network, because the
server has no auth.

## Benchmark your own setup

```bash
pip install -e ".[bench]"           # add ".[bench,quant]" for --with-quant
python bench/run_all.py --ref-audio my_voice.wav --ref-text "Exact transcript."
```

Each configuration runs in its own process. You get `bench/results.md` and a listening page,
`bench/index.html`, with a blind mode.

## Things that didn't work

| tried | result |
|---|---|
| int8 weight-only (torchao) | −0.35 GB more, but slower: RTF 0.79, first chunk 639 ms |
| int4 weight-only (torchao tinygemm) | 1.23 GB idle and fast, but **breaks end-of-speech detection**: some lines run on to the token limit (22.6 s for a 5 s line), even with the output heads kept in bf16 |
| speaker/codec encoders to CPU | no gain: the voice prompt is computed once and cached anyway |
| shorter reference | no gain once the decode context is trimmed |
| GGUF backend (qwentts.cpp) | no Windows wheels for `qwentts-cpp-python` |

int8 and int4 can be tried with `bench/bench.py --quant`. Neither is part of `optimize()`.

## Caveats

- **Pinned to faster-qwen3-tts 0.4.0.** `trim_decode_context` wraps a private method
  (`_prepare_generation`), and `repack_weights` relies on `speech_tokenizer.model`. Re-run the bench
  before upgrading.
- **The 2.5 GB cap fits the 0.6B model.** The 1.7B model, longer references or a larger
  `max_seq_len` need a higher cap or `vram_cap_gb=None`, otherwise you get an OOM.
- Torch backend only, CUDA only.
- The memory savings come from how the model is built, so they carry over to other GPUs and OSes.
  Speed and latency numbers will differ. On Linux, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is an alternative to the cap.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). This is an independent project, not
affiliated with or endorsed by the Qwen team / Alibaba or the faster-qwen3-tts authors. No upstream
code or model weights are included. Qwen3-TTS (Apache-2.0) and faster-qwen3-tts (MIT) are installed
separately under their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Voice cloning:** only clone voices you have the right to use, with the speaker's consent. Don't use
generated speech to impersonate or deceive.
