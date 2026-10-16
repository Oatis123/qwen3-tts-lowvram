"""Benchmark one Qwen3-TTS configuration: VRAM (idle / peak), speed, first-chunk latency.

Run each configuration in its own process (run_all.py does that) so readings don't mix:

    python bench/bench.py --ref-audio voice.wav --ref-text "..." --name stock
    python bench/bench.py --ref-audio voice.wav --ref-text "..." --name optimized --optimize

Writes out/<name>/<line>.wav and out/<name>/stats.json; report.py turns them into a table
and a listening page.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent))
from qwen3_tts_lowvram import (  # noqa: E402
    offload_text_embedding, repack_weights, set_vram_cap, trim_decode_context,
)

LINES = {
    "greet": "Mm. You're back. I was starting to think the chair finally won.",
    "question": "Did you actually sleep last night, or did you just close your eyes "
                "at the monitor for six hours?",
    "long": "Okay, so here's the plan: we fix the bug, you push it at 2:47 AM like a "
            "responsible adult, and tomorrow we pretend this never happened.",
    "soft": "Hey... seriously though. You okay? You've been quiet for a while.",
    "sarcasm": "Wow. Incredible. Truly, a historic achievement in pressing the wrong button.",
}
MiB = 2**20


def gpu_used_mib() -> int | None:
    """Whole-GPU memory.used: catches the CUDA context and anything torch doesn't count."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class PeakMeter(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop_evt = interval, gpu_used_mib(), threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            used = gpu_used_mib()
            if used is not None and (self.peak is None or used > self.peak):
                self.peak = used
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join()
        return self.peak


def quantize(model, kind: str) -> int:
    """Weight-only quantization of the talker + code predictor Linears (torchao), one layer at
    a time so each freed bf16 copy is reused by the next. Returns the number of layers done."""
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    if kind == "int8":
        cfg = Int8WeightOnlyConfig()
    else:  # tinygemm int4 ships with torch; the "plain" layout needs extra kernels
        cfg = Int4WeightOnlyConfig(group_size=128 if kind == "int4" else int(kind.removeprefix("int4g")),
                                   int4_packing_format="tile_packed_to_4d")
    done = 0
    for lin in [m for m in model.model.model.talker.modules() if isinstance(m, torch.nn.Linear)]:
        try:
            quantize_(lin, cfg, filter_fn=lambda m, fqn: m is lin)
            done += 1
        except Exception as e:
            print(f"  keep bf16 {tuple(lin.weight.shape)}: {type(e).__name__}")
    torch.cuda.empty_cache()
    return done


def seam_ratio(wav: np.ndarray, bounds: list[int]) -> float:
    """Worst sample jump at chunk joins vs. the clip's own 99.5th-percentile jump (>~3 = click)."""
    d = np.abs(np.diff(wav))
    ref = np.percentile(d, 99.5) + 1e-9
    jumps = [d[max(0, b - 3):b + 2].max() for b in bounds if 3 < b < len(d) - 2]
    return round(float(max(jumps) / ref), 2) if jumps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--name", required=True, help="output folder / row name")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--language", default="English")
    ap.add_argument("--optimize", action="store_true", help="embedding offload + repack")
    ap.add_argument("--decode-context", type=int, default=0)
    ap.add_argument("--vram-cap", type=float, default=0.0)
    ap.add_argument("--quant", choices=["bf16", "int8", "int4", "int4g64", "int4g32"], default="bf16",
                    help="experimental; int4 broke end-of-speech detection in our tests")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from faster_qwen3_tts import FasterQwen3TTS

    base = gpu_used_mib()
    load_meter = PeakMeter()
    load_meter.start()
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(args.model, max_seq_len=512)
    tricks = args.optimize or args.quant != "bf16"
    if tricks:
        offload_text_embedding(model)
        repack_weights(model)
    n_quant = quantize(model, args.quant) if args.quant != "bf16" else 0
    if args.decode_context:
        trim_decode_context(model, args.decode_context)
    if args.vram_cap:
        set_vram_cap(args.vram_cap)

    def synth(text):
        torch.manual_seed(args.seed)  # same sampling noise for every configuration
        chunks, sr, first, t = [], 24000, None, time.perf_counter()
        for chunk, sr, _ in model.generate_voice_clone_streaming(
            text=text, language=args.language, ref_audio=args.ref_audio, ref_text=args.ref_text,
            chunk_size=8,
        ):
            first = first or time.perf_counter() - t
            chunks.append(chunk)
        if tricks:
            torch.cuda.empty_cache()
        return chunks, sr, first

    synth("Warming up.")  # caches the voice prompt, captures CUDA graphs
    if tricks:
        torch.cuda.empty_cache()
    load_s = time.perf_counter() - t0
    load_meter.stop()

    idle = gpu_used_mib()
    torch.cuda.reset_peak_memory_stats()
    run_meter = PeakMeter()
    run_meter.start()
    out = BENCH_DIR / "out" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line_id, text in LINES.items():
        t = time.perf_counter()
        chunks, sr, first = synth(text)
        gen_s = time.perf_counter() - t
        wav = np.concatenate(chunks).astype(np.float32)
        sf.write(out / f"{line_id}.wav", wav, sr)
        dur = len(wav) / sr
        rows.append({"line": line_id, "audio_s": round(dur, 2), "gen_s": round(gen_s, 3),
                     "rtf": round(gen_s / dur, 3), "first_chunk_ms": round(first * 1000),
                     "seam": seam_ratio(wav, np.cumsum([len(c) for c in chunks])[:-1].tolist())})
        print(f"  [{args.name}] {line_id}: {dur:.1f}s audio in {gen_s:.2f}s, first chunk {first*1000:.0f} ms")
    run_peak = run_meter.stop()

    stats = {
        "name": args.name, "model": args.model, "config": {k: v for k, v in vars(args).items()
                                                            if k not in ("ref_text", "name")},
        "quantized_linears": n_quant, "load_s": round(load_s, 1), "lines": rows,
        "gpu": torch.cuda.get_device_name(0),
        # whole-GPU deltas from before the model loaded = what this process costs
        "idle_mib": None if None in (idle, base) else idle - base,
        "peak_mib": None if None in (run_peak, base) else run_peak - base,
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MiB),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[{args.name}] idle {stats['idle_mib']} MiB, peak {stats['peak_mib']} MiB")


if __name__ == "__main__":
    main()
