"""Convert ASR pipeline outputs into report-ready markdown tables.

Reads:
  - <asr_root>/preds_<system>.jsonl              (one per system, written by eval_wer.py)
  - <asr_root>/pseudo/pseudo_labels_filtered.stats.json
  - <fleurs_test_dir>/refs.jsonl                  (audio -> reference_sw mapping)

Writes:
  - <out_path>: a single markdown file with three sections:
      - Table 11: WER + WER-after-digit-normalization + CER per system
      - Table 12: Filter ablation kept-counts (all 2^3 subsets)
      - Qualitative examples block (first N FLEURS samples, all systems side-by-side)

Digit normalization: KenSpeech transcripts spell digits in word form (e.g. "elfu sita")
while FLEURS references use numerals ("6,000"). To get a fair comparison we run digits
through num2words(lang='sw') in BOTH predictions and references before scoring, and
report both raw and normalized WER.

Usage:
    python whisper_asr/format_results.py \
        --asr_root /home/ec2-user/data/asr_runs \
        --refs_path /home/ec2-user/data/runs/fleurs_sw_ke_test/refs.jsonl \
        --out /home/ec2-user/data/asr_runs/report_tables.md \
        --n_examples 8

Dependency: pip install num2words
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# ---- Normalization ----------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d[\d,]*")


def normalize(text: str) -> str:
    """Same lightweight normalizer as eval_wer.py: lowercase, strip punct, collapse ws."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def _sw_words(n: int) -> str:
    """num2words wrapper with a graceful fallback if Swahili isn't supported."""
    try:
        from num2words import num2words
    except ImportError:
        raise SystemExit("num2words not installed. Run: pip install num2words")
    try:
        return num2words(n, lang="sw")
    except (NotImplementedError, KeyError):
        # Fallback: try English (better than leaving the numeral in mixed text)
        return num2words(n, lang="en")


def digit_normalize(text: str) -> str:
    """Replace runs of digits (allowing thousands-comma) with Swahili word form."""

    def repl(m: re.Match) -> str:
        digits = m.group(0).replace(",", "")
        try:
            return _sw_words(int(digits))
        except Exception:
            return m.group(0)

    return _DIGIT_RUN_RE.sub(repl, text)


# ---- Loading ----------------------------------------------------------------

def load_refs(refs_path: Path) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    with open(refs_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("audio") and e.get("reference_sw"):
                refs[e["audio"]] = e["reference_sw"]
    return refs


def load_systems(asr_root: Path) -> Dict[str, Dict[str, str]]:
    """Returns {system_name: {audio: prediction}}."""
    systems: Dict[str, Dict[str, str]] = {}
    for preds_file in sorted(asr_root.glob("preds_*.jsonl")):
        name = preds_file.stem.replace("preds_", "", 1)
        preds: Dict[str, str] = {}
        with open(preds_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("audio"):
                    preds[e["audio"]] = e.get("prediction", "")
        systems[name] = preds
    return systems


# ---- Scoring ----------------------------------------------------------------

def _filter_pairs(preds: List[str], refs: List[str]) -> Tuple[List[str], List[str]]:
    """Drop pairs with empty references; substitute single space for empty preds."""
    out_p, out_r = [], []
    for p, r in zip(preds, refs):
        if not r.strip():
            continue
        out_p.append(p if p.strip() else " ")
        out_r.append(r)
    return out_p, out_r


def compute_metrics(preds: Dict[str, str], refs: Dict[str, str]):
    """Returns (wer_raw, wer_digit_norm, cer, n_scored)."""
    import evaluate
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    common = sorted(set(preds.keys()) & set(refs.keys()))
    raw_p = [preds[k] for k in common]
    raw_r = [refs[k] for k in common]

    norm_p = [normalize(p) for p in raw_p]
    norm_r = [normalize(r) for r in raw_r]
    dn_p = [normalize(digit_normalize(p)) for p in raw_p]
    dn_r = [normalize(digit_normalize(r)) for r in raw_r]

    p_n, r_n = _filter_pairs(norm_p, norm_r)
    p_d, r_d = _filter_pairs(dn_p, dn_r)

    if not p_n:
        return float("nan"), float("nan"), float("nan"), 0

    wer = 100.0 * wer_metric.compute(predictions=p_n, references=r_n)
    wer_dn = 100.0 * wer_metric.compute(predictions=p_d, references=r_d)
    cer = 100.0 * cer_metric.compute(predictions=p_n, references=r_n)
    return wer, wer_dn, cer, len(p_n)


# ---- Markdown formatting ----------------------------------------------------

# Display order — vanilla baseline first, then trained variants from least to most
SYSTEM_ORDER = [
    "vanilla_small",
    "ft_kenspeech_only",
    "ft_kenspeech_pseudo_raw",
    "ft_kenspeech_pseudo_filtered",
    "ft_kenspeech_gold_upper_bound",
]


def _ordered_systems(systems: Dict[str, Dict[str, str]]) -> List[str]:
    """Return system names in canonical order, with any unknowns appended at the end."""
    known = [n for n in SYSTEM_ORDER if n in systems]
    extras = [n for n in systems if n not in SYSTEM_ORDER]
    return known + extras


def format_wer_table(systems: Dict[str, Dict[str, str]], refs: Dict[str, str]) -> str:
    rows = [
        "| System | WER | WER (digit-norm) | CER | n |",
        "|---|---|---|---|---|",
    ]
    for name in _ordered_systems(systems):
        wer, wer_dn, cer, n = compute_metrics(systems[name], refs)
        display = f"**`{name}`**" if "pseudo_filtered" in name else f"`{name}`"
        rows.append(f"| {display} | {wer:.2f} | {wer_dn:.2f} | {cer:.2f} | {n} |")
    return "\n".join(rows)


# Stats JSON uses keys joined alphabetically, e.g. "confidence+lang_id+repetition"
ABLATION_ORDER = [
    ("none", "(none)"),
    ("confidence", "confidence"),
    ("lang_id", "lang_id"),
    ("repetition", "repetition"),
    ("confidence+lang_id", "confidence + lang_id"),
    ("confidence+repetition", "confidence + repetition"),
    ("lang_id+repetition", "lang_id + repetition"),
    ("confidence+lang_id+repetition", "confidence + lang_id + repetition (all)"),
]


def format_ablation_table(stats_path: Path) -> str:
    if not stats_path.exists():
        return f"_(stats file not found: {stats_path})_"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    abl = stats.get("ablation_kept_by_active_filters", {})
    total = max(1, stats.get("n_total", 1))
    rows = [
        "| Active filters | Pseudo-labels kept | % of total |",
        "|---|---|---|",
    ]
    for key, label in ABLATION_ORDER:
        kept = abl.get(key, 0)
        pct = 100.0 * kept / total
        rows.append(f"| {label} | {kept} | {pct:.1f} |")
    rows.append("")
    rows.append(
        f"_n_total = {stats.get('n_total')}; "
        f"thresholds = {stats.get('thresholds')}_"
    )
    return "\n".join(rows)


def format_examples(
    systems: Dict[str, Dict[str, str]], refs: Dict[str, str], n: int
) -> str:
    """Pick the first n audios common to all systems and emit one markdown block per audio."""
    sys_keys = list(systems.values())
    if not sys_keys:
        return "_(no systems)_"
    common = sorted(set.intersection(*[set(p.keys()) for p in sys_keys]))
    if not common:
        return "_(no audios common to all systems)_"

    blocks = []
    ordered = _ordered_systems(systems)
    for audio in common[:n]:
        rows = [f"**`{audio}`**", "", "| | |", "|---|---|", f"| REF | {refs.get(audio, '')} |"]
        for name in ordered:
            pred = systems[name].get(audio, "")
            rows.append(f"| {name} | {pred} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


# ---- Driver -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_root", default="/home/ec2-user/data/asr_runs")
    parser.add_argument(
        "--refs_path",
        default="/home/ec2-user/data/runs/fleurs_sw_ke_test/refs.jsonl",
    )
    parser.add_argument(
        "--out", default="/home/ec2-user/data/asr_runs/report_tables.md"
    )
    parser.add_argument("--n_examples", type=int, default=8)
    args = parser.parse_args()

    asr_root = Path(args.asr_root)
    refs_path = Path(args.refs_path)
    out_path = Path(args.out)

    refs = load_refs(refs_path)
    print(f"Loaded {len(refs)} references")
    systems = load_systems(asr_root)
    print(f"Loaded {len(systems)} systems: {list(systems.keys())}")
    if not systems:
        raise SystemExit(
            f"No preds_*.jsonl found under {asr_root}. "
            "Did eval_wer.py finish writing predictions?"
        )

    table_wer = format_wer_table(systems, refs)
    table_abl = format_ablation_table(
        asr_root / "pseudo" / "pseudo_labels_filtered.stats.json"
    )
    examples = format_examples(systems, refs, args.n_examples)

    out_text = f"""# ASR Pipeline Results -- Report Tables

These markdown blocks are paste-ready for sections 5.7 and 5.7.1 of the FYP report.
Generated from `{asr_root}` predictions and stats.

---

## Table 11: Swahili ASR Results on FLEURS sw_ke

{table_wer}

---

## Table 12: Pseudo-Label Filter Ablation

{table_abl}

---

## 5.7.2 Qualitative Examples

{examples}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\n--- Preview ---\n")
    print(out_text)


if __name__ == "__main__":
    main()
