"""Cascade text-BLEU eval: ASR transcripts -> NLLB-200 -> BLEU vs FLEURS en_us reference.

Reuses the per-system prediction JSONLs already written by eval_wer.py (so we don't
have to re-run any ASR). For each Sw transcript, we translate to English with
NLLB-200-distilled-1.3B and compute corpus BLEU + chrF against the parallel FLEURS
en_us reference text loaded from refs.jsonl.

This is the "Option B" text-BLEU shortcut for Table 10 of the FYP report. It does NOT
include the MMS-TTS + Whisper-medium re-transcription loop that full ASR-BLEU would
require -- so it isolates the contribution of ASR + MT alone, treating TTS+re-ASR as
a constant that affects all cascade variants equally.

Outputs:
  - <out_dir>/cascade_translations_<system>.jsonl  : per-utterance translations
  - <out_dir>/cascade_bleu.md                       : paste-ready BLEU table

Usage:
    python whisper_asr/eval_cascade.py \
        --asr_root /home/ec2-user/data/asr_runs \
        --refs_path /home/ec2-user/data/runs/fleurs_sw_ke_test/refs.jsonl \
        --out_dir /home/ec2-user/data/asr_runs/cascade_eval

Dependencies: pip install sacrebleu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---- Loading ----------------------------------------------------------------

def load_refs_en(refs_path: Path) -> Dict[str, str]:
    """Returns {audio_filename: english_reference}. Skips entries with empty en ref."""
    refs: Dict[str, str] = {}
    n_skipped_empty = 0
    with open(refs_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            audio = e.get("audio")
            ref_en = (e.get("reference_en") or "").strip()
            if not audio:
                continue
            if not ref_en:
                n_skipped_empty += 1
                continue
            refs[audio] = ref_en
    print(f"Loaded {len(refs)} en references (skipped {n_skipped_empty} with no en parallel)")
    return refs


def load_systems(asr_root: Path) -> Dict[str, Dict[str, str]]:
    """{system_name: {audio: sw_prediction}} from preds_*.jsonl files."""
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
                    preds[e["audio"]] = (e.get("prediction") or "").strip()
        systems[name] = preds
    return systems


# ---- NLLB translation -------------------------------------------------------

@torch.no_grad()
def translate_batch(
    sw_texts: List[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    target_lang_token_id: int,
    device: str,
    max_new_tokens: int = 256,
) -> List[str]:
    """Batch-translate a list of Sw texts to En via NLLB."""
    # Replace empty strings with a single space so the tokenizer doesn't choke
    safe = [t if t.strip() else " " for t in sw_texts]
    enc = tokenizer(safe, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    out = model.generate(
        **enc,
        forced_bos_token_id=target_lang_token_id,
        max_new_tokens=max_new_tokens,
        num_beams=4,  # NLLB default for BLEU-quality decoding
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


# ---- Driver -----------------------------------------------------------------

# Display order — vanilla baseline first, then trained variants from least to most
SYSTEM_ORDER = [
    "vanilla_small",
    "ft_kenspeech_only",
    "ft_kenspeech_pseudo_raw",
    "ft_kenspeech_pseudo_filtered",
    "ft_kenspeech_gold_upper_bound",
]


def _ordered(systems: Dict[str, Dict[str, str]]) -> List[str]:
    known = [n for n in SYSTEM_ORDER if n in systems]
    extras = [n for n in systems if n not in SYSTEM_ORDER]
    return known + extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_root", default="/home/ec2-user/data/asr_runs")
    parser.add_argument(
        "--refs_path",
        default="/home/ec2-user/data/runs/fleurs_sw_ke_test/refs.jsonl",
    )
    parser.add_argument("--out_dir", default="/home/ec2-user/data/asr_runs/cascade_eval")
    parser.add_argument("--mt_model", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--src_lang", default="swh_Latn")
    parser.add_argument("--tgt_lang", default="eng_Latn")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    asr_root = Path(args.asr_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = load_refs_en(Path(args.refs_path))
    if not refs:
        raise SystemExit("No usable en references found; check refs.jsonl.")

    systems = load_systems(asr_root)
    if not systems:
        raise SystemExit(
            f"No preds_*.jsonl found under {asr_root}. Run eval_wer.py first."
        )
    print(f"Loaded {len(systems)} systems: {list(systems.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    print(f"Loading {args.mt_model} on {device} ({args.precision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.mt_model, src_lang=args.src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.mt_model, dtype=dtype).to(device)
    model.eval()
    target_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    # Translate each system's predictions
    name_to_translations: Dict[str, Dict[str, str]] = {}
    for name in _ordered(systems):
        out_path = out_dir / f"cascade_translations_{name}.jsonl"
        # Resume support: skip if already done
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{name}] already translated -> {out_path}, loading...")
            entries: Dict[str, str] = {}
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        entries[e["audio"]] = e["en_translation"]
                    except Exception:
                        continue
            name_to_translations[name] = entries
            continue

        print(f"\n=== Translating system: {name} ===")
        preds = systems[name]
        # Restrict to audios with both an ASR prediction and an en reference
        audios = sorted(set(preds.keys()) & set(refs.keys()))
        translations: Dict[str, str] = {}
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i in range(0, len(audios), args.batch_size):
                chunk = audios[i:i + args.batch_size]
                sw_texts = [preds[a] for a in chunk]
                en_texts = translate_batch(
                    sw_texts, model, tokenizer, target_id, device
                )
                for audio, sw, en in zip(chunk, sw_texts, en_texts):
                    translations[audio] = en
                    out_f.write(json.dumps({
                        "audio": audio,
                        "sw_prediction": sw,
                        "en_translation": en,
                        "en_reference": refs[audio],
                    }, ensure_ascii=False) + "\n")
                if (i // args.batch_size) % 5 == 0:
                    print(f"  {i + len(chunk)}/{len(audios)}  [sw] {sw_texts[0][:50]} -> [en] {en_texts[0][:50]}")
        print(f"  Wrote {len(translations)} translations -> {out_path}")
        name_to_translations[name] = translations

    # Free MT model before scoring
    del model
    torch.cuda.empty_cache()

    # Compute BLEU + chrF
    try:
        import sacrebleu
    except ImportError:
        raise SystemExit("sacrebleu not installed. Run: pip install sacrebleu")

    print("\n=== Scoring ===")
    rows = [
        "| System | BLEU | chrF | n |",
        "|---|---|---|---|",
    ]
    print(f"{'System':<35} {'BLEU':>6} {'chrF':>6}  n")
    print("-" * 55)
    for name in _ordered(name_to_translations):
        translations = name_to_translations[name]
        # Score on the intersection of (translated, has reference)
        common = sorted(set(translations.keys()) & set(refs.keys()))
        hyps = [translations[a] for a in common]
        ref_list = [refs[a] for a in common]
        if not hyps:
            print(f"  {name}: no scorable utterances")
            continue
        bleu = sacrebleu.corpus_bleu(hyps, [ref_list])
        chrf = sacrebleu.corpus_chrf(hyps, [ref_list])
        print(f"{name:<35} {bleu.score:>6.2f} {chrf.score:>6.2f}  {len(hyps)}")
        display = f"**`{name}`**" if "pseudo_raw" in name else f"`{name}`"
        rows.append(f"| {display} | {bleu.score:.2f} | {chrf.score:.2f} | {len(hyps)} |")

    md_out = out_dir / "cascade_bleu.md"
    md_text = (
        "# Cascade Text-BLEU Results (Sw audio -> ASR -> NLLB-200 -> En text)\n\n"
        "Evaluated on the FLEURS sw_ke test set, restricted to utterances with a parallel\n"
        "FLEURS en_us reference. ASR transcripts are reused from `eval_wer.py` outputs;\n"
        "translation uses NLLB-200-distilled-1.3B with beam=4. This is Option B (text-only)\n"
        "of the cascade evaluation -- the full ASR-BLEU with TTS + re-transcription is not\n"
        "performed here.\n\n"
        "## Table 10 (Section 5.6 partial): Cascade Translation Quality\n\n"
        + "\n".join(rows) + "\n"
    )
    md_out.write_text(md_text, encoding="utf-8")
    print(f"\nWrote {md_out}")
    print("\n--- Preview ---\n")
    print(md_text)


if __name__ == "__main__":
    main()
