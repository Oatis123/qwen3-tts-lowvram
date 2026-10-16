"""Summarize bench/out/*/stats.json as a Markdown table and an HTML listening page.

    python bench/report.py [--ref-audio voice.wav]
"""
import argparse
import html
import json
import os
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
esc = html.escape


def load_runs():
    runs = []
    for f in sorted(BENCH_DIR.glob("out/*/stats.json"), key=lambda p: p.stat().st_mtime):
        r = json.loads(f.read_text(encoding="utf-8"))
        lines = r["lines"]
        r["rtf"] = sum(l["rtf"] for l in lines) / len(lines)
        r["first_ms"] = sum(l["first_chunk_ms"] for l in lines) / len(lines)
        r["seam"] = max(l["seam"] for l in lines)
        r["lens"] = " / ".join(f"{l['audio_s']:.1f}" for l in lines)
        runs.append(r)
    return runs


def gb(mib):
    return "?" if mib is None else f"{mib / 1024:.2f} GB"


def markdown(runs) -> str:
    out = ["| config | idle VRAM | peak VRAM | RTF | first chunk | worst seam | line lengths, s |",
           "|---|---|---|---|---|---|---|"]
    for r in runs:
        out.append(f"| {r['name']} | {gb(r['idle_mib'])} | {gb(r['peak_mib'])} | {r['rtf']:.2f} | "
                   f"{r['first_ms']:.0f} ms | {r['seam']} | {r['lens']} |")
    gpu = runs[0]["gpu"] if runs else "?"
    out.append(f"\nGPU: {gpu}. VRAM = whole-GPU memory.used delta from before the model loaded. "
               "RTF = generation time / audio length. Seam > ~3 would mean a click at a chunk join.")
    return "\n".join(out)


def page(runs, ref_audio) -> str:
    rows = "".join(
        f"<tr><td><b>{esc(r['name'])}</b></td><td>{gb(r['idle_mib'])}</td><td>{gb(r['peak_mib'])}</td>"
        f"<td>{r['rtf']:.2f}</td><td>{r['first_ms']:.0f} ms</td><td>{r['seam']}</td></tr>" for r in runs)
    line_ids = [l["line"] for l in runs[0]["lines"]] if runs else []
    sections = ""
    for line_id in line_ids:
        takes = "".join(
            f'<div class=take><span class=name>{esc(r["name"])}</span>'
            f'<audio controls preload=none src="out/{esc(r["name"])}/{line_id}.wav"></audio></div>'
            for r in runs)
        sections += f"<section><h2>{esc(line_id)}</h2>{takes}</section>"
    ref = ""
    if ref_audio:
        rel = os.path.relpath(Path(ref_audio).resolve(), BENCH_DIR).replace(os.sep, "/")
        ref = f'<p><b>Reference</b><br><audio controls preload=none src="{esc(rel)}"></audio></p>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Qwen3-TTS low-VRAM bench</title>
<style>
:root{{--bg:#fff;--fg:#111;--dim:#666;--line:#ddd;--accent:#0a6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--dim:#999;--line:#333;--accent:#5d9}}}}
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:var(--bg);color:var(--fg)}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:.4rem;text-align:left}}
section{{border:1px solid var(--line);border-radius:10px;padding:1rem;margin:1rem 0}} h2{{margin:0 0 .5rem;color:var(--accent)}}
.take{{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;margin:.3rem 0}} .name{{width:200px}}
body.blind .name{{filter:blur(6px)}} body.blind .name:hover{{filter:none}}
</style></head><body><h1>Qwen3-TTS low-VRAM bench</h1>{ref}
<table><tr><th>config</th><th>idle VRAM</th><th>peak VRAM</th><th>RTF</th><th>first chunk</th><th>worst seam</th></tr>{rows}</table>
<p><label><input type=checkbox onchange="document.body.classList.toggle('blind',this.checked)"> blind mode</label></p>
{sections}</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio")
    args = ap.parse_args()
    runs = load_runs()
    md = markdown(runs)
    (BENCH_DIR / "results.md").write_text(md + "\n", encoding="utf-8")
    (BENCH_DIR / "index.html").write_text(page(runs, args.ref_audio), encoding="utf-8")
    print(md)
    print(f"\nListening page: {BENCH_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
