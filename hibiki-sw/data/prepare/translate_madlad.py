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

    def __init__(
        self,
        model_name: str = "google/madlad400-3b-mt",
        device: str = "cuda",
        dtype: str = "float16",
        max_length: int = 256,
        use_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        self.max_length = max_length
        self.use_8bit = use_8bit
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import gc
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Clear any lingering GPU allocations (e.g. Whisper from previous steps)
        gc.collect()
        total_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_gb / 1e9
            total_gb = total_bytes / 1e9
            print(f"GPU memory before load: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

        print(f"Loading {self.model_name} on {self.device} ({'int8' if self.use_8bit else self.dtype})...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.use_8bit:
            # 8-bit quantization: ~6 GB → fits on T4 with room for inference
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            ).to(self.device)

        self._model.eval()

        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            print(f"GPU memory after load:  {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        print("MADLAD-400 loaded.")

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        beam_size: int = 4,
    ) -> List[str]:
        """Translate a batch of texts.

        Args:
            texts: List of source language texts.
            target_lang: Target language code (e.g., "en", "sw").
            beam_size: Beam search width.

        Returns:
            List of translated texts.
        """
        self._load()

        tag = LANG_TAGS.get(target_lang, f"<2{target_lang}>")

        # Prepend language tag to each input
        tagged = [f"{tag} {t}" for t in texts]

        inputs = self._tokenizer(
            tagged,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                num_beams=beam_size,
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate transcriptions using MADLAD-400-3B"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with Whisper transcription JSONs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for translation JSONs")
    parser.add_argument("--source_lang", type=str, required=True,
                        help="Source language code (e.g. sw, en)")
    parser.add_argument("--target_lang", type=str, required=True,
                        help="Target language code (e.g. en, sw)")
    parser.add_argument("--model_name", type=str, default="google/madlad400-3b-mt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--use_8bit", action="store_true",
                        help="Load model in 8-bit (bitsandbytes) — halves VRAM, fits T4 (14.56 GB)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Sentences per batch. Default 8 is safe for T4; raise to 16-32 if GPU has room.")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max token length for translation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Skip this many files (for resuming)")
    args = parser.parse_args()

    translator = MADLADTranslator(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        use_8bit=args.use_8bit,
    )

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
