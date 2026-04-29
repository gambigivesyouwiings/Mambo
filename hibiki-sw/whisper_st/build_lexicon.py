"""Build a Sw<->En bilingual lexicon for prompt augmentation.

Three source modes (combinable):

  1. ALIGNMENTS (preferred): use the (src_word_idx, tgt_word_idx) pairs from
     the data pipeline's alignment step (alignments/sw2en/*.json). For each
     Sw word, we tally which En word it aligns to across the corpus and pick
     the most common alignment as the canonical translation. This is the
     highest-quality source because it preserves sentence-level translation
     context (no per-word NLLB ambiguity).

  2. KENSPEECH + per-word NLLB (fallback): if alignments are not available,
     extract content words from KenSpeech transcripts and translate each
     in isolation with NLLB. Lower quality due to lack of context.

  3. WIKIDATA (augmentation): query Wikidata for Kenya-specific named
     entities (places, people, orgs) with Sw and En labels. Catches proper
     nouns that both NLLB and alignments may miss.

Output: JSONL file `lexicon.jsonl` with one entry per line:
    {"sw": "Nairobi", "en": "Nairobi", "freq": 47, "source": "alignments"}
    {"sw": "mji mkuu", "en": "capital city", "freq": 3, "source": "alignments"}

At inference time, the cascade looks up each word/bigram in the CTC transcript
hypothesis against this lexicon and prepends matched (sw, en) pairs to the
decoder prompt.

Usage:
    # PREFERRED: bidirectional cross-validated lexicon from BOTH directions'
    # alignments. Pairs seen in both directions are flagged as high-confidence.
    python whisper_st/build_lexicon.py \
        --alignments sw2en=/kaggle/working/hibiki-sw/alignments/sw2en \
                     en2sw=/kaggle/input/datasets/<other-acct>/alignments/en2sw \
        --output_path /kaggle/working/lexicon.jsonl \
        --include_wikidata

    # Single direction (no cross-validation)
    python whisper_st/build_lexicon.py \
        --alignments sw2en=/kaggle/working/hibiki-sw/alignments/sw2en \
        --output_path /kaggle/working/lexicon.jsonl

    # Fallback (no alignments yet): per-word NLLB on KenSpeech transcripts
    python whisper_st/build_lexicon.py \
        --kenspeech_dir /kaggle/input/kenspeech-sw \
        --output_path /kaggle/working/lexicon.jsonl \
        --top_n 2000
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Stopword lists
# ---------------------------------------------------------------------------

# Common Swahili function words to exclude from the lexicon (they're not
# content words and don't benefit from prompt augmentation).
SW_STOPWORDS: Set[str] = {
    "na", "ya", "wa", "kwa", "ni", "la", "katika", "za", "cha", "vya",
    "ku", "kuwa", "lakini", "au", "kama", "tu", "pia", "sana", "hii",
    "huu", "ile", "yake", "yangu", "yetu", "wao", "yeye", "mimi", "wewe",
    "sisi", "nyinyi", "ndio", "hapana", "siyo", "wala", "bali", "ila",
    "sasa", "leo", "jana", "kesho", "hapo", "huku", "huko", "pale",
    "akina", "alikuwa", "anakuwa", "atakuwa", "tunakuwa", "wanakuwa",
    "ambaye", "ambao", "ambalo", "kuna", "hakuna", "kupata", "kufanya",
}

EN_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "to", "in",
    "for", "on", "at", "by", "with", "from", "up", "about", "into",
    "through", "during", "this", "that", "these", "those", "it", "and",
    "or", "but", "if", "then", "as", "so",
}

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


# ---------------------------------------------------------------------------
# Step 1 (preferred): extract Sw<->En pairs from pipeline alignments
# ---------------------------------------------------------------------------

def _collect_pairs_from_dir(
    alignments_dir: str,
    direction: str,
    pair_counts: Dict[str, Counter],
    pair_directions: Dict[str, Dict[str, set]],
) -> int:
    """Read one alignment directory and accumulate (sw, en) co-occurrences.

    Direction must be either "sw2en" (source=sw, target=en) or "en2sw"
    (source=en, target=sw). For en2sw files we swap the indices so the
    (sw, en) pair comes out in canonical form.

    Mutates pair_counts and pair_directions in place. Returns number of files
    successfully processed.
    """
    if direction not in ("sw2en", "en2sw"):
        raise ValueError(f"direction must be 'sw2en' or 'en2sw', got {direction!r}")

    align_dir = Path(alignments_dir)
    files = sorted(f for f in align_dir.glob("*.json") if f.name != "index.jsonl")
    print(f"  [{direction}] reading {len(files)} files from {alignments_dir}")
    n_processed = 0

    for jp in files:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        src_text = (data.get("source_text") or "").lower()
        tgt_text = (data.get("translated_text") or "").lower()
        alignment = data.get("alignment") or []

        src_words = src_text.split()
        tgt_words = tgt_text.split()
        if not src_words or not tgt_words or not alignment:
            continue

        for pair in alignment:
            try:
                si, ti = int(pair[0]), int(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            if si < 0 or si >= len(src_words) or ti < 0 or ti >= len(tgt_words):
                continue

            # Canonical orientation: (sw_word, en_word) regardless of file direction
            if direction == "sw2en":
                sw_raw, en_raw = src_words[si], tgt_words[ti]
            else:  # en2sw
                en_raw, sw_raw = src_words[si], tgt_words[ti]

            sw = sw_raw.strip(".,;:!?\"'()")
            en = en_raw.strip(".,;:!?\"'()")
            if not sw or not en or len(sw) <= 2 or len(en) <= 1:
                continue
            if sw in SW_STOPWORDS or en in EN_STOPWORDS:
                continue
            if sw.isdigit() or en.isdigit():
                continue

            pair_counts.setdefault(sw, Counter())[en] += 1
            pair_directions.setdefault(sw, {}).setdefault(en, set()).add(direction)

        n_processed += 1

    return n_processed


def extract_lexicon_from_alignments(
    alignments_specs: List[str],
    min_freq: int = 2,
    min_alignment_consistency: float = 0.4,
) -> List[Dict]:
    """Build a lexicon by aggregating alignment pairs from one or more directions.

    alignments_specs: list of "direction=path" strings, e.g.:
        ["sw2en=/kaggle/working/.../alignments/sw2en",
         "en2sw=/kaggle/input/.../alignments/en2sw"]

    Filtering:
      - Sw word must appear in >= min_freq aligned pairs total (across all dirs)
      - Most-common En target must account for >= min_alignment_consistency of
        the Sw word's alignments

    Bidirectional flag: an entry is `bidirectional=True` if the (sw, en) pair
    was observed in BOTH directions. These are the highest-confidence entries.
    """
    pair_counts: Dict[str, Counter] = {}
    pair_directions: Dict[str, Dict[str, set]] = {}

    for spec in alignments_specs:
        if "=" not in spec:
            raise ValueError(f"Bad alignments spec {spec!r}; expected 'direction=path'")
        direction, path = spec.split("=", 1)
        _collect_pairs_from_dir(path, direction, pair_counts, pair_directions)

    print(f"Aggregated alignments for {len(pair_counts)} unique Sw words")

    entries = []
    for sw, en_counter in pair_counts.items():
        total = sum(en_counter.values())
        if total < min_freq:
            continue
        en, en_count = en_counter.most_common(1)[0]
        consistency = en_count / total
        if consistency < min_alignment_consistency:
            continue
        directions_seen = sorted(pair_directions.get(sw, {}).get(en, set()))
        entries.append({
            "sw": sw,
            "en": en,
            "freq": total,
            "consistency": round(consistency, 3),
            "directions": directions_seen,
            "bidirectional": len(directions_seen) >= 2,
            "source": "alignments",
        })

    n_bidir = sum(1 for e in entries if e["bidirectional"])
    print(f"Kept {len(entries)} entries (freq>={min_freq}, "
          f"consistency>={min_alignment_consistency}); "
          f"{n_bidir} bidirectional (high-confidence)")
    return entries


# ---------------------------------------------------------------------------
# Step 1 (fallback): extract content words from KenSpeech
# ---------------------------------------------------------------------------

def extract_kenspeech_vocabulary(kenspeech_dir: str, min_freq: int = 2) -> Counter:
    """Build a frequency-ranked list of Swahili content words from KenSpeech."""
    import sys
    sys.path.insert(0, str(Path(kenspeech_dir).parent.parent / "data" / "prepare"))
    from kenspeech_loader import KenSpeechLoader

    print(f"Loading KenSpeech from {kenspeech_dir}...")
    ks = KenSpeechLoader(load_audio=False, local_dir=kenspeech_dir)
    print(f"Loaded {len(ks)} samples")

    counts: Counter = Counter()
    for sample in ks:
        text = sample.get("sentence", "").lower()
        words = WORD_RE.findall(text)
        for w in words:
            if w in SW_STOPWORDS or len(w) <= 2 or w.isdigit():
                continue
            counts[w] += 1

    filtered = Counter({w: c for w, c in counts.items() if c >= min_freq})
    print(f"Found {len(filtered)} unique content words (freq >= {min_freq})")
    return filtered


# ---------------------------------------------------------------------------
# Step 2: NLLB-translate each word to English
# ---------------------------------------------------------------------------

def translate_words_to_english(
    words: List[str],
    batch_size: int = 32,
    model_name: str = "facebook/nllb-200-distilled-1.3B",
) -> Dict[str, str]:
    """Translate Swahili words to English in batches via NLLB."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="swh_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")
    translations: Dict[str, str] = {}

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, forced_bos_token_id=forced_bos, max_new_tokens=16, do_sample=False, num_beams=2
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for sw, en in zip(batch, decoded):
            en_clean = en.strip().lower()
            if en_clean and en_clean != sw and en_clean not in EN_STOPWORDS:
                translations[sw] = en_clean
        if (i // batch_size) % 10 == 0:
            print(f"  Translated {min(i + batch_size, len(words))}/{len(words)}")

    return translations


# ---------------------------------------------------------------------------
# Step 3: optional Wikidata augmentation (Kenya named entities)
# ---------------------------------------------------------------------------

def fetch_wikidata_kenya_entities(limit: int = 1000) -> List[Dict[str, str]]:
    """Query Wikidata for Kenya-related entities with Swahili + English labels.

    Returns entries like {"sw": "Mombasa", "en": "Mombasa", "type": "city"}.
    Requires internet access; gracefully returns [] if the query fails.
    """
    try:
        import requests
    except ImportError:
        print("requests not installed; skipping Wikidata augmentation")
        return []

    sparql = f"""
    SELECT DISTINCT ?item ?swLabel ?enLabel ?typeLabel WHERE {{
      ?item wdt:P17 wd:Q114 .
      ?item rdfs:label ?swLabel .
      ?item rdfs:label ?enLabel .
      OPTIONAL {{ ?item wdt:P31 ?type . ?type rdfs:label ?typeLabel . FILTER(LANG(?typeLabel) = "en") }}
      FILTER(LANG(?swLabel) = "sw")
      FILTER(LANG(?enLabel) = "en")
    }}
    LIMIT {limit}
    """
    url = "https://query.wikidata.org/sparql"
    print(f"Querying Wikidata for up to {limit} Kenya entities...")
    try:
        resp = requests.get(
            url,
            params={"query": sparql, "format": "json"},
            headers={"User-Agent": "hibiki-sw-lexicon/0.1"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Wikidata query failed: {e}")
        return []

    entries = []
    for binding in data.get("results", {}).get("bindings", []):
        sw = binding.get("swLabel", {}).get("value", "").strip().lower()
        en = binding.get("enLabel", {}).get("value", "").strip().lower()
        type_label = binding.get("typeLabel", {}).get("value", "").strip().lower()
        if sw and en:
            entries.append({"sw": sw, "en": en, "type": type_label or "entity"})
    print(f"Got {len(entries)} entries from Wikidata")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", nargs="+", default=None,
                        help="PREFERRED: one or more direction=path entries, e.g. "
                             "'sw2en=/path/to/sw2en en2sw=/path/to/en2sw'. "
                             "Pairs observed in both directions are flagged bidirectional.")
    parser.add_argument("--kenspeech_dir", type=str, default=None,
                        help="FALLBACK: local KenSpeech dir for per-word NLLB translation. "
                             "Used only when --alignments_dir is not set.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL path for the lexicon")
    parser.add_argument("--top_n", type=int, default=2000,
                        help="Per-word NLLB mode: translate top N most frequent words")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Minimum frequency for a word to be included")
    parser.add_argument("--min_alignment_consistency", type=float, default=0.4,
                        help="Alignment mode: drop entries where the most-common En "
                             "translation accounts for less than this fraction of alignments")
    parser.add_argument("--include_wikidata", action="store_true",
                        help="Augment with Kenya-specific named entities from Wikidata")
    parser.add_argument("--wikidata_limit", type=int, default=1000)
    args = parser.parse_args()

    if not args.alignments and not args.kenspeech_dir:
        parser.error("Provide either --alignments (preferred) or --kenspeech_dir (fallback)")

    entries: List[Dict] = []

    if args.alignments:
        # Step 1A: alignments-based extraction (preferred)
        entries = extract_lexicon_from_alignments(
            args.alignments,
            min_freq=args.min_freq,
            min_alignment_consistency=args.min_alignment_consistency,
        )
    else:
        # Step 1B: per-word NLLB translation (fallback)
        counts = extract_kenspeech_vocabulary(args.kenspeech_dir, min_freq=args.min_freq)
        top_words = [w for w, _ in counts.most_common(args.top_n)]
        sw_en = translate_words_to_english(top_words)
        for sw, en in sw_en.items():
            entries.append({"sw": sw, "en": en, "freq": counts[sw], "source": "kenspeech"})

    # Step 2: optional Wikidata augmentation
    if args.include_wikidata:
        seen_sw = {e["sw"] for e in entries}
        for wd in fetch_wikidata_kenya_entities(limit=args.wikidata_limit):
            if wd["sw"] not in seen_sw:
                entries.append({
                    "sw": wd["sw"], "en": wd["en"], "freq": 0,
                    "source": f"wikidata:{wd.get('type','entity')}",
                })
                seen_sw.add(wd["sw"])

    # Sort by frequency (descending) for nicer browsing
    entries.sort(key=lambda e: (-e.get("freq", 0), e["sw"]))

    # Write
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_bidir = sum(1 for e in entries if e.get("bidirectional"))
    print(f"\nDone! {len(entries)} lexicon entries ({n_bidir} bidirectional) -> {args.output_path}")
    print("First 10 entries:")
    for e in entries[:10]:
        bi = "*" if e.get("bidirectional") else " "
        dirs = ",".join(e.get("directions", [])) or e.get("source", "")
        print(f"  {bi} {e['sw']:20s} -> {e['en']:30s} (freq={e['freq']}, dirs={dirs})")


if __name__ == "__main__":
    main()
