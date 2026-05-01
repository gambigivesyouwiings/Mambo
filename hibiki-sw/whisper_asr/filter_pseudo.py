"""Filter raw pseudo-labels with three composable, ablatable filters.

Filters (each can be turned on/off via CLI, all on by default):
  1. CONFIDENCE     : avg_log_prob > confidence_threshold  (default -1.0 ≈ perplexity ~2.7)
  2. REPETITION     : n-gram-repetition ratio < repetition_threshold (catches "( ( ( (" loops)
  3. LANGUAGE_ID    : detected language is Sw with high confidence  (catches hallucinated En)

Outputs:
  - <out>.jsonl                       : kept entries
  - <out>.dropped.jsonl               : dropped entries with rejection reason(s) for inspection
  - <out>.stats.json                  : ablation stats (kept counts under each filter combination)

Ablation stats are computed for every filter SUBSET (2^N combinations) so the paper can
report "what would have been kept under filter X alone, X+Y, ..." without re-running.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


# ---- Individual filter checks -------------------------------------------------

def _passes_confidence(entry: Dict, threshold: float) -> bool:
    """avg_log_prob is per-token log probability. Higher = more confident.
    -1.0 ≈ perplexity 2.7; -2.0 ≈ perplexity 7.4. Anything below ~-2.5 is suspicious.
    """
    return entry.get("avg_log_prob", -math.inf) > threshold


def _ngram_repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are duplicates. 0.0 = all unique, 1.0 = all the same.

    Catches both 'a a a a a' (high 1-gram repetition) and 'the cat the cat the cat'
    (high 2-gram repetition). Use the MAX over n=1..5 as the score.
    """
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    n_repeated = sum(c for c in counts.values() if c > 1)
    return n_repeated / len(grams)


def _max_repetition_score(text: str) -> float:
    return max(_ngram_repetition_ratio(text, n) for n in (1, 2, 3, 4, 5))


def _passes_repetition(entry: Dict, threshold: float) -> bool:
    return _max_repetition_score(entry.get("pseudo_label", "")) < threshold


# Lazy-load language detector (heavy dep)
_LID = None
def _get_lang_detector():
    global _LID
    if _LID is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError as e:
            raise RuntimeError(
                "language-id filter requires `lingua-language-detector`. "
                "Install with: pip install lingua-language-detector"
            ) from e
        # Restrict to languages we plausibly might see — speeds up detection
        # and reduces false positives from low-resource neighbors.
        _LID = LanguageDetectorBuilder.from_languages(
            Language.SWAHILI, Language.ENGLISH, Language.FRENCH, Language.ARABIC
        ).build()
    return _LID


def _passes_lang_id(entry: Dict, sw_confidence_threshold: float) -> bool:
    text = entry.get("pseudo_label", "").strip()
    if not text:
        return False
    detector = _get_lang_detector()
    confidences = detector.compute_language_confidence_values(text)
    sw_conf = 0.0
    for cv in confidences:
        # lingua API: cv has .language and .value
        if cv.language.name == "SWAHILI":
            sw_conf = cv.value
            break
    return sw_conf >= sw_confidence_threshold


# ---- Combine filters and compute ablation stats -------------------------------

FILTERS = ["confidence", "repetition", "lang_id"]


def _evaluate_entry(
    entry: Dict,
    confidence_threshold: float,
    repetition_threshold: float,
    sw_confidence_threshold: float,
    skip_lang_id: bool = False,
) -> Set[str]:
    """Returns the SET of filter names the entry FAILS. Empty set = passes all."""
    failed = set()
    if not _passes_confidence(entry, confidence_threshold):
        failed.add("confidence")
    if not _passes_repetition(entry, repetition_threshold):
        failed.add("repetition")
    if not skip_lang_id:
        try:
            if not _passes_lang_id(entry, sw_confidence_threshold):
                failed.add("lang_id")
        except RuntimeError:
            # If lingua isn't installed, treat lang_id as always-pass and warn once.
            pass
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="pseudo_labels.jsonl from pseudo_label.py")
    parser.add_argument("--output", required=True,
                        help="Filtered output prefix; writes .jsonl, .dropped.jsonl, .stats.json")

    parser.add_argument("--confidence_threshold", type=float, default=-1.0,
                        help="Min avg_log_prob to keep. Default -1.0 ≈ perplexity 2.7.")
    parser.add_argument("--repetition_threshold", type=float, default=0.5,
                        help="Max n-gram repetition ratio to keep. Default 0.5.")
    parser.add_argument("--sw_confidence_threshold", type=float, default=0.7,
                        help="Min Sw lang-id confidence to keep. Default 0.7.")
    parser.add_argument("--skip_lang_id", action="store_true",
                        help="Disable the lang-id filter (e.g. if lingua isn't installed).")
    parser.add_argument("--use_gold_when_available", action="store_true",
                        help="For entries that have gold_label, replace pseudo_label with gold "
                             "in the kept output. Used to build the 'gold upper-bound' training set.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_path = out_path.with_suffix(".jsonl")
    dropped_path = out_path.with_suffix(".dropped.jsonl")
    stats_path = out_path.with_suffix(".stats.json")

    n_total = 0
    fail_counts = Counter()  # singletons
    fail_combo_counts = Counter()  # frozensets
    by_source = Counter()
    kept_by_source = Counter()

    with open(in_path, "r", encoding="utf-8") as in_f, \
         open(kept_path, "w", encoding="utf-8") as kept_f, \
         open(dropped_path, "w", encoding="utf-8") as dropped_f:
        for line in in_f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            n_total += 1
            by_source[entry.get("source", "?")] += 1

            failed = _evaluate_entry(
                entry,
                confidence_threshold=args.confidence_threshold,
                repetition_threshold=args.repetition_threshold,
                sw_confidence_threshold=args.sw_confidence_threshold,
                skip_lang_id=args.skip_lang_id,
            )

            for f in failed:
                fail_counts[f] += 1
            fail_combo_counts[frozenset(failed)] += 1

            if not failed:
                if args.use_gold_when_available and entry.get("gold_label"):
                    entry = {**entry, "pseudo_label": entry["gold_label"], "label_source": "gold"}
                else:
                    entry = {**entry, "label_source": "pseudo"}
                kept_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept_by_source[entry.get("source", "?")] += 1
            else:
                entry["_rejected_by"] = sorted(failed)
                dropped_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Ablation: kept counts under every filter SUBSET (which filters are active)
    ablation = {}
    for r in range(0, len(FILTERS) + 1):
        for subset in combinations(FILTERS, r):
            active = set(subset)
            kept = sum(
                cnt for failed_set, cnt in fail_combo_counts.items()
                if failed_set.isdisjoint(active)  # entry kept iff none of its failures are in active
            )
            ablation["+".join(sorted(subset)) or "none"] = kept

    stats = {
        "input": str(in_path),
        "n_total": n_total,
        "n_kept": ablation["+".join(FILTERS)],
        "by_source_total": dict(by_source),
        "by_source_kept_full_filter": dict(kept_by_source),
        "fail_counts": dict(fail_counts),
        "ablation_kept_by_active_filters": ablation,
        "thresholds": {
            "confidence": args.confidence_threshold,
            "repetition": args.repetition_threshold,
            "sw_lang_id": args.sw_confidence_threshold,
        },
        "use_gold_when_available": args.use_gold_when_available,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Total: {n_total}")
    print(f"Kept (all filters on): {stats['n_kept']}  "
          f"({100.0 * stats['n_kept'] / max(1, n_total):.1f}%)")
    print(f"Per-filter rejection counts: {dict(fail_counts)}")
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
