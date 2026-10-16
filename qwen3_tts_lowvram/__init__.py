"""Cut Qwen3-TTS VRAM roughly in half without touching the voice or the weights' precision.

Works on a loaded ``faster_qwen3_tts.FasterQwen3TTS`` (torch backend), before its first
generation (CUDA graphs are captured on the first call)::

    from faster_qwen3_tts import FasterQwen3TTS
    from qwen3_tts_lowvram import optimize

    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", max_seq_len=512)
    optimize(model)

Relies on faster-qwen3-tts internals (tested with 0.4.0): re-run the bench after upgrading.
"""
from __future__ import annotations

import torch

__all__ = [
    "CpuEmbedding",
    "offload_text_embedding",
    "repack_weights",
    "trim_decode_context",
    "set_vram_cap",
    "optimize",
]
__version__ = "0.1.0"


class CpuEmbedding(torch.nn.Module):
    """Embedding table kept in RAM: rows are looked up on CPU, only the few result vectors
    are shipped to the GPU."""

    def __init__(self, emb: torch.nn.Embedding, device: torch.device | str):
        super().__init__()
        self.emb = emb.to("cpu")
        self.device = device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids.to("cpu")).to(self.device, non_blocking=True)


def _inner(model):
    return model.model.model  # Qwen3TTSForConditionalGeneration


def offload_text_embedding(model) -> None:
    """Move the talker's text embedding (151936 x 2048 bf16 = 594 MiB) to CPU.

    It is only used to build the prompt; the CUDA-graph decode loop runs on codec
    embeddings, so the CPU lookup costs microseconds once per line."""
    talker_model = _inner(model).talker.model
    if not isinstance(talker_model.text_embedding, CpuEmbedding):
        talker_model.text_embedding = CpuEmbedding(talker_model.text_embedding, model.device)


def repack_weights(model) -> None:
    """Re-upload every GPU weight so they sit in fresh, tightly packed allocator segments.

    Freeing one tensor (e.g. the offloaded embedding) leaves a hole inside a segment that
    other weights still pin, so ``empty_cache`` can't return it. The codec
    (``speech_tokenizer``) is not a registered submodule and has to be collected separately."""
    inner = _inner(model)
    modules = [inner, inner.speech_tokenizer.model]
    tensors = {id(t): t for m in modules for t in [*m.parameters(), *m.buffers()] if t.is_cuda}
    for t in tensors.values():
        t.data = t.data.cpu()
    torch.cuda.empty_cache()
    for t in tensors.values():
        t.data = t.data.to(model.device)


def trim_decode_context(model, frames: int = 25) -> None:
    """Decode only the last ``frames`` reference frames as codec context on the first chunks.

    In voice-clone (ICL) mode every line starts by running the *whole* reference
    (~12.5 frames/s, so ~210 frames for 17 s) through the codec decoder, which is where the
    per-line VRAM spike and part of the first-chunk latency come from. ``_prepare_generation``
    hands those codes out only for that decode; the talker prompt is already built from the
    full cached reference, so the voice is unchanged. 25 frames = 2 s of acoustic context."""
    if getattr(model, "_lowvram_decode_context", None) is not None:
        model._lowvram_decode_context = frames
        return
    original = model._prepare_generation
    model._lowvram_decode_context = frames

    def patched(*args, **kwargs):
        *rest, ref_codes = original(*args, **kwargs)
        if ref_codes is not None:
            ref_codes = ref_codes[-model._lowvram_decode_context:]
        return (*rest, ref_codes)

    model._prepare_generation = patched


def set_vram_cap(gb: float, device: int = 0) -> None:
    """Hard per-process ceiling. When the caching allocator would grow past it, it frees
    cached blocks and retries instead (a stand-in for ``expandable_segments``, which is not
    available on Windows). Raises OOM only if the real working set doesn't fit."""
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / total), device)


def optimize(model, *, decode_context: int | None = 25, vram_cap_gb: float | None = 2.5) -> None:
    """Apply everything: embedding offload + repack, decode-context trim, VRAM cap.

    Call before the first generation. Pass ``None`` to skip a step; raise ``vram_cap_gb``
    (or disable it) for bigger models or longer references. Also call
    ``torch.cuda.empty_cache()`` between lines to hand spare blocks back."""
    offload_text_embedding(model)
    repack_weights(model)
    if decode_context:
        trim_decode_context(model, decode_context)
    if vram_cap_gb:
        set_vram_cap(vram_cap_gb)
