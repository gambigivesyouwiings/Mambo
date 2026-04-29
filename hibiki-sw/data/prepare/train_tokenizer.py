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
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def download_text_data(
    output_file: str,
    num_sentences: int = 5_000_000,
    cv_dataset_dirs: dict = None,
    cv_split: str = "validated",
    use_kenspeech: bool = False,
):
    """Download and combine English + Swahili text data.

    Args:
        output_file: Path to write combined text file
        num_sentences: Target number of sentences
        cv_dataset_dirs: Dict mapping lang code to local CV directory path,
            e.g. {"sw": "/content/cv-corpus-19.0-2024-09-13/sw",
                   "en": "/content/cv-corpus-19.0-2024-09-13/en"}
        cv_split: Common Voice split to use ("validated", "train", etc.)
        use_kenspeech: If True, load Swahili text from KenSpeech instead of CV
    """
    from datasets import load_dataset

    print("Collecting text data...")
    texts = []

    # KenSpeech Swahili transcripts
    if use_kenspeech:
        from data.prepare.kenspeech_loader import KenSpeechLoader
        print("  Loading KenSpeech Swahili transcripts...")
        try:
            ks = KenSpeechLoader(load_audio=False)
            count = 0
            for sentence in ks.text_iterator():
                texts.append(sentence)
                count += 1
                if count >= num_sentences // 6:
                    break
            print(f"    Loaded {count} sentences")
        except Exception as e:
            print(f"  Warning: Could not load KenSpeech: {e}")

    # Common Voice transcripts (from local dataset)
    if cv_dataset_dirs:
        from data.prepare.local_cv_loader import CommonVoiceLocal
        for lang, dataset_dir in cv_dataset_dirs.items():
            print(f"  Loading Common Voice ({lang}) from {dataset_dir}...")
            try:
                cv = CommonVoiceLocal(
                    dataset_dir=dataset_dir, split=cv_split, load_audio=False
                )
                count = 0
                for sentence in cv.text_iterator():
                    texts.append(sentence)
                    count += 1
                    if count >= num_sentences // 6:
                        break
                print(f"    Loaded {count} sentences")
            except Exception as e:
                print(f"  Warning: Could not load Common Voice {lang}: {e}")
    elif not use_kenspeech:
        print("  Skipping Common Voice (no --cv_dataset_dir provided)")

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
            ds = load_dataset(
                "google/fleurs",
                lang,
                split="train",
                trust_remote_code=True,
            )
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

    if args.text_file and os.path.exists(args.text_file):
        text_file = args.text_file
    else:
        text_file = os.path.join(args.output_dir, "training_text.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        download_text_data(text_file, args.num_sentences,
                           cv_dataset_dirs=cv_dirs, cv_split=args.cv_split,
                           use_kenspeech=args.kenspeech)

    train_sentencepiece(text_file, args.output_dir, args.vocab_size)


if __name__ == "__main__":
    main()
