"""Tokenize text data for Stage 1 text pretraining.

Reads text from Common Voice transcripts, OPUS parallel data, and CC-100,
tokenizes with the trained SentencePiece model, and saves as .npy files.

Usage:
    python data/prepare/tokenize_text.py \
        --tokenizer_model tokenizer/sp_ensw_32k.model \
        --output_dir /path/to/text_tokens \
        --max_samples 500000
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def load_text_sources(
    max_samples: int = 500_000,
    cv_dataset_dirs: dict = None,
    cv_split: str = "validated",
    use_kenspeech: bool = False,
) -> List[str]:
    """Load text data from multiple sources.

    Args:
        max_samples: Maximum total number of text samples
        cv_dataset_dirs: Dict mapping lang code to local CV directory path,
            e.g. {"sw": "/content/cv-corpus-19.0-2024-09-13/sw",
                   "en": "/content/cv-corpus-19.0-2024-09-13/en"}
        cv_split: Common Voice split to use ("validated", "train", etc.)
        use_kenspeech: If True, load Swahili text from KenSpeech
    """
    from datasets import load_dataset

    texts = []

    # KenSpeech Swahili transcripts
    if use_kenspeech:
        from data.prepare.kenspeech_loader import KenSpeechLoader
        print("Loading KenSpeech Swahili transcripts...")
        try:
            ks = KenSpeechLoader(load_audio=False)
            count = 0
            for sentence in ks.text_iterator():
                if len(sentence) > 10:
                    texts.append(sentence)
                    count += 1
                if count >= max_samples // 4:
                    break
            print(f"  Loaded {count} sentences")
        except Exception as e:
            print(f"  Warning: Could not load KenSpeech: {e}")

    # Common Voice transcripts (from local dataset)
    if cv_dataset_dirs:
        from data.prepare.local_cv_loader import CommonVoiceLocal
        for lang, dataset_dir in cv_dataset_dirs.items():
            print(f"Loading Common Voice transcripts ({lang}) from {dataset_dir}...")
            try:
                cv = CommonVoiceLocal(
                    dataset_dir=dataset_dir, split=cv_split, load_audio=False
                )
                count = 0
                for sentence in cv.text_iterator():
                    if len(sentence) > 10:
                        texts.append(sentence)
                        count += 1
                    if count >= max_samples // 4:
                        break
                print(f"  Loaded {count} sentences")
            except Exception as e:
                print(f"  Warning: {e}")
    elif not use_kenspeech:
        print("Skipping Common Voice (no --cv_dataset_dir provided)")

    # OPUS parallel data
    print("Loading OPUS CCAligned (en-sw)...")
    try:
        ds = load_dataset(
            "opus/CCAligned", lang1="en", lang2="sw",
            split="train", streaming=True,
        )
        count = 0
        for sample in ds:
            en_text = sample["translation"]["en"].strip()
            sw_text = sample["translation"]["sw"].strip()
            if len(en_text) > 10:
                texts.append(en_text)
            if len(sw_text) > 10:
                texts.append(sw_text)
            count += 2
            if count >= max_samples // 2:
                break
    except Exception as e:
        print(f"  Warning: {e}")

    print(f"Total text samples: {len(texts)}")
    return texts


def tokenize_and_save(
    texts: List[str],
    sp_model: spm.SentencePieceProcessor,
    output_dir: str,
    seq_length: int = 1024,
):
    """Tokenize texts and save as concatenated chunks."""
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize all texts and concatenate
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        ids = sp_model.encode(text, out_type=int)
        all_tokens.extend(ids)

    all_tokens = np.array(all_tokens, dtype=np.int32)
    print(f"Total tokens: {len(all_tokens):,}")

    # Split into chunks of seq_length
    num_chunks = len(all_tokens) // seq_length
    print(f"Creating {num_chunks} chunks of length {seq_length}")

    for i in tqdm(range(num_chunks), desc="Saving chunks"):
        chunk = all_tokens[i * seq_length : (i + 1) * seq_length]
        out_path = os.path.join(output_dir, f"text_{i:07d}.npy")
        np.save(out_path, chunk)

    print(f"Saved {num_chunks} token files to {output_dir}")
    return num_chunks


def main():
    parser = argparse.ArgumentParser(description="Tokenize text for Stage 1")
    parser.add_argument("--tokenizer_model", type=str, required=True,
                        help="Path to SentencePiece .model file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=500_000)
    parser.add_argument("--cv_dataset_dir", type=str, action="append", default=None,
                        help="Common Voice local dir as lang:path, e.g. "
                             "sw:/content/cv-corpus-19.0-2024-09-13/sw "
                             "(can be specified multiple times for multiple languages)")
    parser.add_argument("--cv_split", type=str, default="validated",
                        help="Common Voice split to use (default: validated)")
    parser.add_argument("--kenspeech", action="store_true",
                        help="Include KenSpeech Swahili transcripts")
    args = parser.parse_args()

    # Parse --cv_dataset_dir entries into a dict
    cv_dirs = None
    if args.cv_dataset_dir:
        cv_dirs = {}
        for entry in args.cv_dataset_dir:
            lang, path = entry.split(":", 1)
            cv_dirs[lang] = path

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)
    print(f"Loaded tokenizer: vocab_size={sp.get_piece_size()}")

    texts = load_text_sources(args.max_samples, cv_dataset_dirs=cv_dirs,
                              cv_split=args.cv_split,
                              use_kenspeech=args.kenspeech)
    tokenize_and_save(texts, sp, args.output_dir, args.seq_length)


if __name__ == "__main__":
    main()
