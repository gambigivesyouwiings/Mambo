"""Render a cascade demo directory as a single HTML page with audio players.

Reads transcripts.jsonl produced by cascade_demo.py and emits an index.html
in the same directory with HTML5 <audio> controls for every source + target
WAV. Open the HTML in any browser to listen — no server needed.

Usage (Windows PowerShell):

    python hibiki-sw\\whisper_asr\\cascade_demo_to_html.py C:\\Users\\User\\Downloads\\asr_results\\sw2en

The script writes <demo_dir>/index.html and (optionally) opens it.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path


HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cascade S2S Demo — {direction}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1   {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2   {{ margin-top: 2.2em; padding-top: 0.4em; border-top: 1px solid #ccc; }}
  .sample {{ background: #fafafa; border: 1px solid #e0e0e0; padding: 1em 1.4em;
             border-radius: 6px; margin-bottom: 1.5em; }}
  .row    {{ display: grid; grid-template-columns: 200px 1fr; gap: 0.6em 1em;
             margin: 0.6em 0; align-items: center; }}
  .label  {{ font-weight: 600; color: #555; font-size: 0.9em; }}
  .text   {{ font-size: 1em; line-height: 1.4em; }}
  .gold   {{ background: #fff7cc; padding: 0.4em 0.6em; border-radius: 4px; }}
  .ref    {{ background: #d8f1d8; padding: 0.4em 0.6em; border-radius: 4px; }}
  .variant {{ border-left: 3px solid #4a90e2; padding-left: 0.8em; margin: 0.5em 0; }}
  audio   {{ width: 360px; }}
  .name   {{ font-family: Consolas, monospace; color: #4a90e2; font-weight: 600; }}
  details {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
             padding: 0.6em 0.9em; margin-bottom: 0.6em; }}
  summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
<h1>Cascade S2S Demo — {direction}</h1>
<p>Each sample shows the source audio, the gold reference, and one block per
ASR variant containing the variant's transcript, MT output, and synthesised
target audio. {n_samples} samples, {n_variants} ASR variants.</p>
"""

FOOT = """
</body>
</html>
"""


def render(demo_dir: Path, open_browser: bool = True) -> Path:
    transcripts_path = demo_dir / "transcripts.jsonl"
    if not transcripts_path.is_file():
        raise SystemExit(f"transcripts.jsonl not found in {demo_dir}")

    samples = [json.loads(l) for l in transcripts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not samples:
        raise SystemExit(f"transcripts.jsonl in {demo_dir} is empty")

    direction = demo_dir.name
    variants = list(samples[0]["asr"].keys())

    parts = [HEAD.format(direction=html.escape(direction),
                         n_samples=len(samples),
                         n_variants=len(variants))]

    for s in samples:
        idx = s["idx"]
        parts.append(f'<div class="sample"><h2>Sample {idx:02d}'
                     + (f' &nbsp;<small>(fleurs_id={html.escape(str(s.get("fleurs_id")))})</small>' if s.get("fleurs_id") else '')
                     + '</h2>')

        # Source row
        src = s["source_wav"]
        parts.append('<div class="row"><div class="label">Source audio</div>'
                     f'<audio controls preload="none" src="{html.escape(src)}"></audio></div>')
        if s.get("source_text_gold"):
            parts.append('<div class="row"><div class="label">Source gold transcript</div>'
                         f'<div class="text gold">{html.escape(s["source_text_gold"])}</div></div>')
        if s.get("reference_target"):
            parts.append('<div class="row"><div class="label">Target reference</div>'
                         f'<div class="text ref">{html.escape(s["reference_target"])}</div></div>')

        # Per-variant blocks
        for name in variants:
            asr_text = s["asr"].get(name, "")
            mt_text  = s["mt"].get(name, "")
            tgt_path = s["target_wavs"].get(name, "")
            parts.append('<div class="variant">')
            parts.append(f'<p><span class="name">[{html.escape(name)}]</span></p>')
            parts.append('<div class="row"><div class="label">ASR transcript</div>'
                         f'<div class="text">{html.escape(asr_text)}</div></div>')
            parts.append('<div class="row"><div class="label">MT output</div>'
                         f'<div class="text">{html.escape(mt_text)}</div></div>')
            if tgt_path:
                parts.append('<div class="row"><div class="label">Synthesised target</div>'
                             f'<audio controls preload="none" src="{html.escape(tgt_path)}"></audio></div>')
            parts.append('</div>')

        parts.append('</div>')

    parts.append(FOOT)

    out_path = demo_dir / "index.html"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")
    if open_browser:
        webbrowser.open(out_path.as_uri())
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("demo_dir", type=Path,
                   help="Cascade demo directory (containing transcripts.jsonl + source/ + target/)")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open in browser")
    args = p.parse_args()

    demo_dir = args.demo_dir.resolve()
    if not demo_dir.is_dir():
        raise SystemExit(f"{demo_dir} is not a directory")
    render(demo_dir, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
