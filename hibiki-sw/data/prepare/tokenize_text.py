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


def load_text_sources(max_samples: int = 500_000) -> List[str]:
    """Load text data from multiple sources."""
    from datasets import load_dataset

    texts = []

    # Common Voice transcripts
    for lang in ["en", "sw"]:
        print(f"Loading Common Voice transcripts ({lang})...")
        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_16_0",
                lang, split="train", streaming=True,
            )
            count = 0
            for sample in ds:
                sent = sample["sentence"].strip()
                if len(sent) > 10:  # skip very short sentences
                    texts.append(sent)
                    count += 1
                if count >= max_samples // 4:
                    break
        except Exception as e:
            print(f"  Warning: {e}")

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
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)
    print(f"Loaded tokenizer: vocab_size={sp.get_piece_size()}")

    texts = load_text_sources(args.max_samples)
    tokenize_and_save(texts, sp, args.output_dir, args.seq_length)


if __name__ == "__main__":
    main()
