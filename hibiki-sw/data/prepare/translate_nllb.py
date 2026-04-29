"""Translate transcriptions using NLLB-200-distilled-1.3B.

Drop-in replacement for translate_madlad.py. Same JSON I/O contract, same
index.jsonl format, same resume logic — downstream alignment/synthesis
cells do not need to change.

Why NLLB instead of MADLAD-400-3B:
    MADLAD-400-3B is a T5 variant. In our Kaggle 2xT4 environment it produced
    degenerate outputs across every dtype we tried (int8, NF4, fp16, bf16,
    fp32) and the minimal HF model-card repro also failed. The suspected
    cause is a T5x->HF weight-conversion fragility in recent transformers
    versions. NLLB-200-distilled-1.3B is NOT T5-based, has native swh_Latn
    <-> eng_Latn support, fits on one T4 in fp16 (~2.6 GB weights), and
    uses forced_bos_token_id for target-language selection instead of
    MADLAD's <2xx> prompt tags.

Usage:
    # Swahili -> English
    python data/prepare/translate_nllb.py \
        --input_dir /kaggle/working/hibiki-sw/transcriptions/sw \
        --output_dir /kaggle/working/hibiki-sw/translations/sw2en \
        --source_lang sw --target_lang en --batch_size 16

    # English -> Swahili
    python data/prepare/translate_nllb.py \
        --input_dir /kaggle/working/hibiki-sw/transcriptions/en \
        --output_dir /kaggle/working/hibiki-sw/translations/en2sw \
        --source_lang en --target_lang sw --batch_size 16

    # Smoke test only (no input_dir/output_dir needed)
    python data/prepare/translate_nllb.py \
        --source_lang sw --target_lang en --smoke_test
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# NLLB-200 Translation
# ---------------------------------------------------------------------------

# NLLB uses BCP-47-style language codes as vocabulary tokens.
# swh_Latn = Swahili (Kenya/Tanzania/Uganda variant, Latin script).
NLLB_CODES = {
    "en": "eng_Latn",
    "sw": "swh_Latn",
    "swh": "swh_Latn",
}


class NLLBTranslator:
    """Batch translator using NLLB-200-distilled-1.3B."""

    _DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-1.3B",
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
        # Initial src_lang is a placeholder; translate_batch resets it per call.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="eng_Latn")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        self._model.eval()

        if torch.cuda.is_available():
            self._log_gpu_mem("after load ")
        print("NLLB-200 loaded.")

    @staticmethod
    def _log_gpu_mem(label: str):
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i} {label}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    def _resolve_code(self, lang: str) -> str:
        code = NLLB_CODES.get(lang)
        if code is None:
            raise ValueError(
                f"Unknown language {lang!r}. NLLB codes available: {sorted(set(NLLB_CODES.values()))}. "
                f"Add a mapping to NLLB_CODES if you need a new language."
            )
        return code

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str,
    ) -> List[str]:
        """Translate a batch of texts.

        Args:
            texts: Source-language texts.
            target_lang: Target language code (e.g., "en", "sw").
            source_lang: Source language code (e.g., "sw", "en"). NLLB needs
                this on the tokenizer side (MADLAD did not).

        Returns:
            List of translated strings.
        """
        self._load()

        src_code = self._resolve_code(source_lang)
        tgt_code = self._resolve_code(target_lang)

        # NLLB selects source language via the tokenizer (prepends the src code
        # token to encoder input) and target language via forced_bos_token_id
        # on the decoder side.
        self._tokenizer.src_lang = src_code

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._model.device)

        forced_bos = self._tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=self.max_length,
                do_sample=False,
                num_beams=4,
            )

        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate_single(self, text: str, target_lang: str, source_lang: str) -> str:
        return self.translate_batch([text], target_lang, source_lang)[0]


# ---------------------------------------------------------------------------
# Pipeline: process transcription JSONs
# ---------------------------------------------------------------------------

def translate_transcriptions(
    input_dir: str,
    output_dir: str,
    translator: NLLBTranslator,
    source_lang: str,
    target_lang: str,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    resume_from: int = 0,
):
    """Translate all transcription JSONs in input_dir.

    Same JSON schema as translate_madlad.py so downstream steps are unchanged.
    """
    input_dir = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(
        f for f in input_dir.glob("*.json")
        if f.name != "index.jsonl"
    )

    if resume_from > 0:
        json_files = json_files[resume_from:]
    if max_samples:
        json_files = json_files[:max_samples]

    print(f"Found {len(json_files)} transcription files to translate")
    print(f"Direction: {source_lang} -> {target_lang}")

    index_path = os.path.join(output_dir, "index.jsonl")
    index_mode = "a" if resume_from > 0 else "w"

    processed = 0
    errors = 0
    start_time = time.time()

    batch_files = []
    batch_texts = []

    def _flush():
        nonlocal processed, errors
        if not batch_texts:
            return
        try:
            translations = translator.translate_batch(batch_texts, target_lang, source_lang)
        except Exception as e:
            errors += len(batch_texts)
            if errors <= 10:
                print(f"  Batch error: {e}")
            batch_files.clear()
            batch_texts.clear()
            return

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

    with open(index_path, index_mode, encoding="utf-8") as idx_f:
        for json_path in tqdm(json_files, desc=f"Translating {source_lang}->{target_lang}"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    transcription = json.load(f)

                text = transcription.get("text", "").strip()
                if not text:
                    continue

                batch_files.append((json_path, transcription))
                batch_texts.append(text)

                if len(batch_texts) >= batch_size:
                    _flush()

                    if processed % 500 == 0 and processed > 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed
                        print(f"  Translated: {processed} | Errors: {errors} | "
                              f"Rate: {rate:.1f}/s | Elapsed: {elapsed/60:.1f}min")

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error on {json_path.name}: {e}")
                continue

        _flush()

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


def run_smoke_test(translator: NLLBTranslator, source_lang: str, target_lang: str):
    """Translate 5 canonical sentences and print results for visual inspection."""
    samples = SMOKE_SAMPLES.get(source_lang)
    if samples is None:
        raise ValueError(f"No smoke samples for source_lang={source_lang!r}.")

    print(f"\n=== Smoke test: {source_lang} -> {target_lang} ({len(samples)} sentences) ===")

    translator._load()

    # Sanity check: both NLLB codes must resolve to real vocab tokens.
    src_code = translator._resolve_code(source_lang)
    tgt_code = translator._resolve_code(target_lang)
    unk_id = translator._tokenizer.unk_token_id
    src_id = translator._tokenizer.convert_tokens_to_ids(src_code)
    tgt_id = translator._tokenizer.convert_tokens_to_ids(tgt_code)
    print(f"Lang sanity: src={src_code!r}->{src_id}  tgt={tgt_code!r}->{tgt_id}  (unk={unk_id})")
    if src_id == unk_id or tgt_id == unk_id:
        print("ERROR: language code resolved to <unk>. Tokenizer/model mismatch.")
        return

    results = translator.translate_batch(samples, target_lang, source_lang)
    print()
    for src, tgt in zip(samples, results):
        print(f"  [{source_lang}] {src}")
        print(f"  [{target_lang}] {tgt}")
        print()

    unique_tokens = {tok for tgt in results for tok in tgt.split()}
    if len(unique_tokens) <= 3:
        print("WARNING: output looks degenerate (<=3 unique tokens across all translations).")
        print("Do NOT run the full pipeline.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Translate transcriptions using NLLB-200")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory with Whisper transcription JSONs (not required with --smoke_test)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for translation JSONs (not required with --smoke_test)")
    parser.add_argument("--source_lang", type=str, required=True,
                        help="Source language code (e.g. sw, en)")
    parser.add_argument("--target_lang", type=str, required=True,
                        help="Target language code (e.g. en, sw)")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Inference dtype. NLLB is stable in fp16 (unlike T5/MADLAD). "
                             "Use float32 only for debugging; bfloat16 is fine on GPUs with native bf16.")
    parser.add_argument("--device_map", type=str, default="auto",
                        help='HF device_map. "auto" shards if needed; on a single T4 the 2.6 GB '
                             'weights fit comfortably and it will load on cuda:0.')
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Sentences per batch. 16 is comfortable for NLLB-1.3B on a single T4.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Skip this many files (for resuming)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Translate 5 canonical sentences and print output; skip the full pipeline.")
    args = parser.parse_args()

    translator = NLLBTranslator(
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
