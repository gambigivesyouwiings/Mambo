"""Translate transcriptions using MADLAD-400-3B.

Takes Whisper transcription JSONs as input and produces translations
for each utterance. Supports both En→Sw and Sw→En directions.

Designed for Colab (free T4 GPU) — NOT counted against Kaggle GPU quota.

Usage:
    # Translate Swahili transcriptions to English
    python data/prepare/translate_madlad.py \
        --input_dir /content/drive/MyDrive/hibiki-sw/transcriptions/sw \
        --output_dir /content/drive/MyDrive/hibiki-sw/translations/sw2en \
        --source_lang sw \
        --target_lang en \
        --batch_size 32

    # Translate English transcriptions to Swahili
    python data/prepare/translate_madlad.py \
        --input_dir /content/drive/MyDrive/hibiki-sw/transcriptions/en \
        --output_dir /content/drive/MyDrive/hibiki-sw/translations/en2sw \
        --source_lang en \
        --target_lang sw \
        --batch_size 32
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MADLAD-400 Translation
# ---------------------------------------------------------------------------

# MADLAD-400 language tags
LANG_TAGS = {
    "en": "<2en>",
    "sw": "<2sw>",
    "swh": "<2sw>",  # alias
}


class MADLADTranslator:
    """Batch translator using MADLAD-400-3B."""

    _DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    def __init__(
        self,
        model_name: str = "google/madlad400-3b-mt",
        device: str = "cuda",
        dtype: str = "float16",
        max_length: int = 256,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        if dtype not in self._DTYPE_MAP:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {list(self._DTYPE_MAP)}")
        self.dtype = self._DTYPE_MAP[dtype]
        self.max_length = max_length
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import gc
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_gpu_mem("before load")

        print(f"Loading {self.model_name} ({self.dtype}, device_map={self.device_map!r})...")
        # use_fast=False: MADLAD's fast tokenizer splits <2xx> language tags into
        # subword pieces (<, 2, en, >), so the model never sees the target-language
        # instruction and hallucinates random languages. Slow tokenizer keeps them
        # as single vocab tokens, matching the model card's recommendation.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # No quantization: bitsandbytes int8/NF4 both produce degenerate outputs on
        # T5/MADLAD (see kaggle_v7/v8 logs). fp16 weights (~12 GB) shard cleanly
        # across 2× T4 via device_map="auto" — each GPU holds ~6 GB with room for
        # activations. For single-GPU, pass device_map="cuda:0" or leave "auto".
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        self._model.eval()

        if torch.cuda.is_available():
            self._log_gpu_mem("after load ")
        print("MADLAD-400 loaded.")

    @staticmethod
    def _log_gpu_mem(label: str):
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i} {label}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
    ) -> List[str]:
        """Translate a batch of texts.

        Args:
            texts: List of source language texts.
            target_lang: Target language code (e.g., "en", "sw").

        Returns:
            List of translated texts.
        """
        self._load()

        tag = LANG_TAGS.get(target_lang, f"<2{target_lang}>")

        # Prepend language tag. MADLAD model card uses greedy generation; beam
        # search in bf16 on T5 can enter degenerate length-normalised loops that
        # emit the same token (or wrong-language spam) until max_new_tokens.
        tagged = [f"{tag} {t}" for t in texts]

        inputs = self._tokenizer(
            tagged,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                do_sample=False,
            )

        translations = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translations

    def translate_single(self, text: str, target_lang: str) -> str:
        """Translate a single text."""
        results = self.translate_batch([text], target_lang)
        return results[0]


# ---------------------------------------------------------------------------
# Pipeline: process transcription JSONs
# ---------------------------------------------------------------------------

def translate_transcriptions(
    input_dir: str,
    output_dir: str,
    translator: MADLADTranslator,
    source_lang: str,
    target_lang: str,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    resume_from: int = 0,
):
    """Translate all transcription JSONs in input_dir.

    For each input JSON (from transcribe_whisper.py), produces an output JSON
    containing the original transcription, its translation, and word-level
    info needed for downstream alignment.

    Args:
        input_dir: Directory with Whisper transcription JSONs.
        output_dir: Output directory for translation JSONs.
        translator: MADLADTranslator instance.
        source_lang: Source language code.
        target_lang: Target language code.
        batch_size: Number of sentences to translate at once.
        max_samples: Max files to process.
        resume_from: Skip this many files (for resuming).
    """
    input_dir = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Find all transcription JSONs (exclude index.jsonl)
    json_files = sorted(
        f for f in input_dir.glob("*.json")
        if f.name != "index.jsonl"
    )

    if resume_from > 0:
        json_files = json_files[resume_from:]
    if max_samples:
        json_files = json_files[:max_samples]

    print(f"Found {len(json_files)} transcription files to translate")
    print(f"Direction: {source_lang} → {target_lang}")

    # Process in batches
    index_path = os.path.join(output_dir, "index.jsonl")
    index_mode = "a" if resume_from > 0 else "w"

    processed = 0
    errors = 0
    start_time = time.time()

    # Collect batches
    batch_files = []
    batch_texts = []

    with open(index_path, index_mode, encoding="utf-8") as idx_f:
        for json_path in tqdm(json_files, desc=f"Translating {source_lang}→{target_lang}"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    transcription = json.load(f)

                text = transcription.get("text", "").strip()
                if not text:
                    continue

                batch_files.append((json_path, transcription))
                batch_texts.append(text)

                # Translate when batch is full
                if len(batch_texts) >= batch_size:
                    try:
                        translations = translator.translate_batch(batch_texts, target_lang)
                    except Exception as e:
                        errors += len(batch_texts)
                        if errors <= 10:
                            print(f"  Batch error: {e}")
                        batch_files.clear()
                        batch_texts.clear()
                        continue

                    for (jp, trans_data), translation in zip(batch_files, translations):
                        result = {
                            "source_file": jp.name,
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "source_text": trans_data["text"],
                            "translated_text": translation,
                            "source_words": trans_data.get("words", []),
                            "duration": trans_data.get("duration", 0),
                            "audio_duration": trans_data.get("audio_duration", 0),
                            "sample_idx": trans_data.get("sample_idx", -1),
                            "original_sentence": trans_data.get("original_sentence", ""),
                        }

                        out_name = jp.stem + f"_{target_lang}.json"
                        out_path = os.path.join(output_dir, out_name)
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)

                        idx_entry = {
                            "file": out_name,
                            "source_file": jp.name,
                            "source_text": result["source_text"][:200],
                            "translated_text": result["translated_text"][:200],
                            "duration": result["audio_duration"],
                        }
                        idx_f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")
                        processed += 1

                    batch_files.clear()
                    batch_texts.clear()

                    # Progress
                    if processed % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed
                        print(f"  Translated: {processed} | Errors: {errors} | "
                              f"Rate: {rate:.1f}/s | Elapsed: {elapsed/60:.1f}min")

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error on {json_path.name}: {e}")
                continue

        # Flush remaining batch
        if batch_texts:
            try:
                translations = translator.translate_batch(batch_texts, target_lang)
                for (jp, trans_data), translation in zip(batch_files, translations):
                    result = {
                        "source_file": jp.name,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "source_text": trans_data["text"],
                        "translated_text": translation,
                        "source_words": trans_data.get("words", []),
                        "duration": trans_data.get("duration", 0),
                        "audio_duration": trans_data.get("audio_duration", 0),
                        "sample_idx": trans_data.get("sample_idx", -1),
                        "original_sentence": trans_data.get("original_sentence", ""),
                    }
                    out_name = jp.stem + f"_{target_lang}.json"
                    out_path = os.path.join(output_dir, out_name)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    idx_entry = {
                        "file": out_name,
                        "source_file": jp.name,
                        "source_text": result["source_text"][:200],
                        "translated_text": result["translated_text"][:200],
                        "duration": result["audio_duration"],
                    }
                    idx_f.write(json.dumps(idx_entry, ensure_ascii=False) + "\n")
                    processed += 1
            except Exception as e:
                errors += 1
                print(f"  Error on final batch: {e}")

    elapsed = time.time() - start_time
    print(f"\nDone! Translated: {processed} | Errors: {errors} | "
          f"Time: {elapsed/60:.1f}min")
    return processed


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

SMOKE_SAMPLES = {
    "sw": [
        "Habari yako?",
        "Nimefurahi kukutana nawe.",
        "Wao walikuwa hawana hata habari walicheza pamoja.",
        "Tafadhali nisaidie na kazi hii.",
        "Leo ni siku nzuri ya kusoma vitabu.",
    ],
    "en": [
        "Hello, how are you?",
        "I am very happy to meet you.",
        "Please help me with this task.",
        "Today is a good day for reading books.",
        "The children were playing together in the garden.",
    ],
}


def run_smoke_test(translator: MADLADTranslator, source_lang: str, target_lang: str):
    """Translate a handful of canonical sentences and print the results.

    Used to confirm the model is producing coherent output (not "t t t t..." tokens)
    before kicking off a multi-hour full run.
    """
    samples = SMOKE_SAMPLES.get(source_lang)
    if samples is None:
        raise ValueError(f"No smoke samples for source_lang={source_lang!r}. Add to SMOKE_SAMPLES.")

    print(f"\n=== Smoke test: {source_lang} -> {target_lang} ({len(samples)} sentences) ===")

    # Tokenization sanity check: <2xx> must be ONE token, not split into <,2,xx,>
    translator._load()
    tag = LANG_TAGS.get(target_lang, f"<2{target_lang}>")
    probe = f"{tag} {samples[0]}"
    tokens = translator._tokenizer.tokenize(probe)
    tag_id = translator._tokenizer.convert_tokens_to_ids(tag)
    unk_id = translator._tokenizer.unk_token_id
    print(f"Tag sanity: tag={tag!r} -> id={tag_id} (unk={unk_id})")
    print(f"First 8 tokens of {probe!r}: {tokens[:8]}")
    if tag_id == unk_id or tag not in tokens:
        print(f"ERROR: language tag {tag!r} is not a single vocab token. "
              f"Model will hallucinate target languages. Fix the tokenizer loading.")
        return

    results = translator.translate_batch(samples, target_lang)
    print()
    for src, tgt in zip(samples, results):
        print(f"  [{source_lang}] {src}")
        print(f"  [{target_lang}] {tgt}")
        print()

    unique_tokens = {tok for tgt in results for tok in tgt.split()}
    if len(unique_tokens) <= 3:
        print("WARNING: output looks degenerate (<=3 unique tokens across all translations).")
        print("This indicates the quantization/dtype path is broken. Do NOT run the full pipeline.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate transcriptions using MADLAD-400-3B"
    )
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory with Whisper transcription JSONs (not required with --smoke_test)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for translation JSONs (not required with --smoke_test)")
    parser.add_argument("--source_lang", type=str, required=True,
                        help="Source language code (e.g. sw, en)")
    parser.add_argument("--target_lang", type=str, required=True,
                        help="Target language code (e.g. en, sw)")
    parser.add_argument("--model_name", type=str, default="google/madlad400-3b-mt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Inference dtype. bf16 is the T5/MADLAD native dtype and the only "
                             "one that produces coherent output (fp16 overflows T5 attention softmax). "
                             "On T4, bf16 compute is emulated via fp32 — ~2-3x slower than fp16 but stable.")
    parser.add_argument("--device_map", type=str, default="auto",
                        help='HF device_map. "auto" shards across all visible GPUs; "cuda:0" pins to GPU 0.')
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Sentences per batch. Default 8 is safe for T4; raise to 16-32 if GPU has room.")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max token length for translation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Skip this many files (for resuming)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Translate 5 canonical sentences and print output; skip the full pipeline.")
    args = parser.parse_args()

    translator = MADLADTranslator(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        device_map=args.device_map,
    )

    if args.smoke_test:
        run_smoke_test(translator, args.source_lang, args.target_lang)
        return

    if not args.input_dir or not args.output_dir:
        parser.error("--input_dir and --output_dir are required unless --smoke_test is set.")

    translate_transcriptions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        translator=translator,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
