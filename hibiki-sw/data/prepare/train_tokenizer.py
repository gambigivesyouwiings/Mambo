"""Train a SentencePiece BPE tokenizer on English + Swahili text.

Sources:
    - CC-100 (en, sw)
    - Common Voice transcripts (en, sw)
    - OPUS parallel corpora (en-sw)

Produces a 32k BPE model at tokenizer/sp_ensw_32k.model

Usage:
    python data/prepare/train_tokenizer.py \
        --output_dir tokenizer \
        --vocab_size 32000 \
        --num_sentences 5000000
"""

import argparse
import os
import tempfile
from pathlib import Path


def download_text_data(output_file: str, num_sentences: int = 5_000_000):
    """Download and combine English + Swahili text data."""
    from datasets import load_dataset

    print("Downloading text data...")
    texts = []

    # Common Voice transcripts
    for lang in ["en", "sw"]:
        print(f"  Loading Common Voice ({lang})...")
        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_16_0",
                lang, split="train", streaming=True,
            )
            count = 0
            for sample in ds:
                texts.append(sample["sentence"])
                count += 1
                if count >= num_sentences // 6:
                    break
        except Exception as e:
            print(f"  Warning: Could not load Common Voice {lang}: {e}")

    # OPUS (CCAligned en-sw)
    print("  Loading OPUS CCAligned (en-sw)...")
    try:
        ds = load_dataset(
            "opus/CCAligned", lang1="en", lang2="sw",
            split="train", streaming=True,
        )
        count = 0
        for sample in ds:
            texts.append(sample["translation"]["en"])
            texts.append(sample["translation"]["sw"])
            count += 2
            if count >= num_sentences // 3:
                break
    except Exception as e:
        print(f"  Warning: Could not load OPUS: {e}")

    # FLEURS transcripts
    for lang in ["en_us", "sw_ke"]:
        print(f"  Loading FLEURS ({lang})...")
        try:
            ds = load_dataset("google/fleurs", lang, split="train")
            for sample in ds:
                texts.append(sample["transcription"])
        except Exception as e:
            print(f"  Warning: Could not load FLEURS {lang}: {e}")

    # Write to file
    print(f"  Writing {len(texts)} sentences to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for t in texts:
            line = t.strip()
            if line:
                f.write(line + "\n")

    return len(texts)


def train_sentencepiece(input_file: str, output_dir: str, vocab_size: int):
    """Train a SentencePiece BPE model."""
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, f"sp_ensw_{vocab_size // 1000}k")

    print(f"Training SentencePiece BPE model (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=os.cpu_count(),
        byte_fallback=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        train_extremely_large_corpus=True,
        max_sentence_length=4096,
    )

    print(f"Model saved to {model_prefix}.model")
    print(f"Vocab saved to {model_prefix}.vocab")

    # Quick validation
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    test_en = "The weather is nice today."
    test_sw = "Hali ya hewa ni nzuri leo."
    print(f"\nTest tokenization:")
    print(f"  EN: {test_en}")
    print(f"  -> {sp.encode(test_en, out_type=str)}")
    print(f"  SW: {test_sw}")
    print(f"  -> {sp.encode(test_sw, out_type=str)}")

    return f"{model_prefix}.model"


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--output_dir", type=str, default="tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_sentences", type=int, default=5_000_000)
    parser.add_argument("--text_file", type=str, default=None,
                        help="Skip download, use existing text file")
    args = parser.parse_args()

    if args.text_file and os.path.exists(args.text_file):
        text_file = args.text_file
    else:
        text_file = os.path.join(args.output_dir, "training_text.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        download_text_data(text_file, args.num_sentences)

    train_sentencepiece(text_file, args.output_dir, args.vocab_size)


if __name__ == "__main__":
    main()
