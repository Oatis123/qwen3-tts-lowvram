"""Run the standard comparison (one process per configuration), then build the report.

    python bench/run_all.py --ref-audio voice.wav --ref-text "Exact transcript."
    python bench/run_all.py --ref-audio voice.wav --ref-text-file voice.txt --with-quant
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

CONFIGS = {
    "stock": [],
    "offload-repack": ["--optimize"],
    "offload-repack-ctx25": ["--optimize", "--decode-context", "25"],
    "optimized": ["--optimize", "--decode-context", "25", "--vram-cap", "2.5"],  # = optimize()
}
QUANT_CONFIGS = {
    "optimized-int8": ["--optimize", "--decode-context", "25", "--vram-cap", "2.5", "--quant", "int8"],
    "optimized-int4": ["--optimize", "--decode-context", "25", "--vram-cap", "2.5", "--quant", "int4"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--ref-audio", required=True)
ap.add_argument("--ref-text")
ap.add_argument("--ref-text-file")
ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ap.add_argument("--with-quant", action="store_true", help="also run the experimental int8/int4 rows")
args = ap.parse_args()
ref_text = args.ref_text or Path(args.ref_text_file).read_text(encoding="utf-8").strip()

shutil.rmtree(BENCH_DIR / "out", ignore_errors=True)
(BENCH_DIR / "logs").mkdir(exist_ok=True)
configs = {**CONFIGS, **(QUANT_CONFIGS if args.with_quant else {})}
for name, flags in configs.items():
    print(f"=== {name}")
    log = BENCH_DIR / "logs" / f"{name}.log"
    cmd = [sys.executable, str(BENCH_DIR / "bench.py"), "--name", name, "--model", args.model,
           "--ref-audio", args.ref_audio, "--ref-text", ref_text, *flags]
    with open(log, "w", encoding="utf-8") as f:
        ok = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode == 0
    print("    ok" if ok else f"    FAILED, see {log}")

subprocess.run([sys.executable, str(BENCH_DIR / "report.py"), "--ref-audio", args.ref_audio])
