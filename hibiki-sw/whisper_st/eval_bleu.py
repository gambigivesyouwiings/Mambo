"""Evaluate transcript-prompted Whisper ST on FLEURS sw_ke test set.

Computes BLEU (sacrebleu) between predicted English translations and FLEURS
ground-truth English references. Designed to support side-by-side comparison
of multiple system variants:

    - vanilla:           Whisper-base/-small/-large-v3 zero-shot ST
    - finetuned:         our model, no lexicon
    - finetuned+lex:     our model with lexicon-augmented prompts

Usage:
    # 1) Generate predictions with each variant (using inference.py)
    # 2) Run this to score them all against the same FLEURS reference
    python whisper_st/eval_bleu.py \
        --references_path /kaggle/working/fleurs_sw_ke_test_refs.jsonl \
        --predictions vanilla=preds_vanilla.jsonl \
                      finetuned=preds_finetuned.jsonl \
                      finetuned_lex=preds_finetuned_lex.jsonl

The references file format:
    {"audio": "fleurs_sw_ke_00001.wav", "reference_en": "..."}

Predictions file format (matches inference.py output):
    {"audio": "fleurs_sw_ke_00001.wav", "translation": "..."}

This script also prints a small qualitative table for the first N examples so
you can eyeball where each variant succeeds/fails.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index_by_audio(entries: List[dict], key: str) -> Dict[str, str]:
    return {e["audio"]: e.get(key, "") for e in entries}


def compute_bleu(refs: List[str], hyps: List[str]) -> float:
    """Compute corpus BLEU via sacrebleu."""
    try:
        import sacrebleu
    except ImportError:
        raise SystemExit("sacrebleu not installed. Run: pip install sacrebleu")
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score


def chrf(refs: List[str], hyps: List[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references_path", type=str, required=True,
                        help="JSONL with {audio, reference_en}")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="One or more system_name=path.jsonl entries")
    parser.add_argument("--show_examples", type=int, default=10)
    args = parser.parse_args()

    refs_all = load_jsonl(args.references_path)
    refs_by_audio = index_by_audio(refs_all, "reference_en")
    print(f"Loaded {len(refs_by_audio)} references")

    systems = {}
    for spec in args.predictions:
        if "=" not in spec:
            raise SystemExit(f"Bad spec {spec!r}; expected name=path")
        name, path = spec.split("=", 1)
        preds = load_jsonl(path)
        preds_by_audio = index_by_audio(preds, "translation")
        systems[name] = preds_by_audio
        print(f"  {name}: {len(preds_by_audio)} predictions from {path}")

    # Align: only score audios present in references AND every system
    common_audios = set(refs_by_audio.keys())
    for name, preds_by_audio in systems.items():
        common_audios &= set(preds_by_audio.keys())
    common_audios = sorted(common_audios)
    print(f"\nScoring on {len(common_audios)} audios common to all systems\n")

    refs = [refs_by_audio[a] for a in common_audios]

    print(f"{'System':<25s} {'BLEU':>8s} {'chrF':>8s}")
    print("-" * 45)
    rows = []
    for name, preds_by_audio in systems.items():
        hyps = [preds_by_audio[a] for a in common_audios]
        b = compute_bleu(refs, hyps)
        c = chrf(refs, hyps)
        print(f"{name:<25s} {b:>8.2f} {c:>8.2f}")
        rows.append((name, b, c, hyps))

    # Qualitative table
    if args.show_examples > 0:
        print(f"\n=== First {args.show_examples} examples ===\n")
        for i in range(min(args.show_examples, len(common_audios))):
            audio = common_audios[i]
            print(f"[{audio}]")
            print(f"  REF: {refs[i]}")
            for name, _, _, hyps in rows:
                print(f"  {name:<20s}: {hyps[i]}")
            print()


if __name__ == "__main__":
    main()
