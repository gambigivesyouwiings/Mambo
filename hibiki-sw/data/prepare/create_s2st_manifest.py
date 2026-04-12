"""Create TSV manifests for Stage 3/4 speech translation training.

Combines parallel audio pairs (source en audio + target sw audio) with
aligned text tokens into a manifest file. The manifest format is:

    source_audio_path\ttarget_audio_path\ttext_path\tvoice_category

Data sources:
    - FLEURS en_us <-> sw_ke (pre-encoded Mimi tokens)
    - CoVoST2 en->sw (if available)
    - Synthetic pairs via VITS TTS (Stage 0)

Usage:
    python data/prepare/create_s2st_manifest.py \
        --source_token_dir /path/to/en_tokens \
        --target_token_dir /path/to/sw_tokens \
        --text_token_dir /path/to/aligned_text \
        --output_manifest /path/to/manifest.tsv \
        --direction en2sw
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def create_fleurs_manifest(
    en_token_dir: str,
    sw_token_dir: str,
    text_token_dir: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
    max_frames: int = 250,
):
    """Create manifest from FLEURS en_us <-> sw_ke parallel data.

    FLEURS provides parallel utterances. We match by utterance ID
    and create aligned text tokens using the target transcription.
    """
    from datasets import load_dataset

    print("Loading FLEURS metadata...")
    en_ds = load_dataset("google/fleurs", "en_us", split="train")
    sw_ds = load_dataset("google/fleurs", "sw_ke", split="train")

    # Build mapping: sentence_id -> index for matching
    en_by_id = {sample["id"]: i for i, sample in enumerate(en_ds)}
    sw_by_id = {sample["id"]: i for i, sample in enumerate(sw_ds)}

    # Find matching IDs
    common_ids = set(en_by_id.keys()) & set(sw_by_id.keys())
    print(f"Found {len(common_ids)} parallel utterance pairs")

    # Load tokenizer for text alignment
    sp = None
    if tokenizer_path:
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)

    os.makedirs(text_token_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    entries = []
    for uid in tqdm(sorted(common_ids), desc="Creating manifest"):
        en_idx = en_by_id[uid]
        sw_idx = sw_by_id[uid]

        # Find token files
        en_pattern = f"fleurs_en_us_train_{en_idx:05d}.npy"
        sw_pattern = f"fleurs_sw_ke_train_{sw_idx:05d}.npy"

        en_token_path = os.path.join(en_token_dir, en_pattern)
        sw_token_path = os.path.join(sw_token_dir, sw_pattern)

        if not os.path.exists(en_token_path) or not os.path.exists(sw_token_path):
            continue

        # Check frame count
        try:
            en_tokens = np.load(en_token_path, mmap_mode="r")
            sw_tokens = np.load(sw_token_path, mmap_mode="r")
        except Exception:
            continue

        if en_tokens.shape[1] > max_frames or sw_tokens.shape[1] > max_frames:
            continue

        # Create aligned text tokens
        text_path = os.path.join(text_token_dir, f"text_{uid:05d}.npy")
        if sp and not os.path.exists(text_path):
            sw_text = sw_ds[sw_idx]["transcription"]
            text_ids = sp.encode(sw_text, out_type=int)

            # Create frame-aligned text: spread text tokens across audio frames
            T = sw_tokens.shape[1]
            aligned = create_aligned_text_tokens(text_ids, T)
            np.save(text_path, aligned.astype(np.int32))

        if not os.path.exists(text_path):
            continue

        # Voice category (default: neutral=2)
        voice_cat = 2

        entries.append((en_token_path, sw_token_path, text_path, voice_cat))

    # Write manifest
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# source_audio\ttarget_audio\ttext\tvoice_category\n")
        for src, tgt, txt, vc in entries:
            f.write(f"{src}\t{tgt}\t{txt}\t{vc}\n")

    print(f"Written {len(entries)} entries to {output_path}")
    return len(entries)


def create_aligned_text_tokens(
    text_ids: List[int],
    num_frames: int,
    epad_id: int = 3,
) -> np.ndarray:
    """Distribute text tokens across audio frames.

    Creates a (num_frames,) array where text tokens are evenly spread
    across frames, with EPAD tokens filling frames without new text.

    This is a simple uniform distribution. For better alignment, use the
    contextual alignment algorithm (contextual_align.py) with forced
    alignment timestamps.
    """
    aligned = np.full(num_frames, epad_id, dtype=np.int32)

    if len(text_ids) == 0:
        return aligned

    # Spread tokens evenly
    if len(text_ids) >= num_frames:
        # More text tokens than frames: truncate
        aligned[:] = text_ids[:num_frames]
    else:
        # Fewer tokens: distribute evenly
        step = num_frames / len(text_ids)
        for i, tid in enumerate(text_ids):
            frame_idx = min(int(i * step), num_frames - 1)
            aligned[frame_idx] = tid

    return aligned


def create_directory_manifest(
    source_dir: str,
    target_dir: str,
    text_dir: str,
    output_path: str,
    voice_category: int = 2,
):
    """Create manifest from pre-organized directories.

    Expects matching filenames across source_dir, target_dir, and text_dir.
    """
    source_files = sorted(Path(source_dir).glob("*.npy"))
    entries = []

    for src_path in tqdm(source_files, desc="Scanning files"):
        stem = src_path.stem
        tgt_path = Path(target_dir) / f"{stem}.npy"
        txt_path = Path(text_dir) / f"{stem}.npy"

        if tgt_path.exists() and txt_path.exists():
            entries.append((str(src_path), str(tgt_path), str(txt_path), voice_category))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# source_audio\ttarget_audio\ttext\tvoice_category\n")
        for src, tgt, txt, vc in entries:
            f.write(f"{src}\t{tgt}\t{txt}\t{vc}\n")

    print(f"Written {len(entries)} entries to {output_path}")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Create S2ST manifest")
    parser.add_argument("--source", type=str, default="fleurs",
                        choices=["fleurs", "kenspeech", "directory"],
                        help="Data source type")
    parser.add_argument("--source_token_dir", type=str, required=True,
                        help="Directory with source audio .npy tokens")
    parser.add_argument("--target_token_dir", type=str, required=True,
                        help="Directory with target audio .npy tokens")
    parser.add_argument("--text_token_dir", type=str, required=True,
                        help="Directory for aligned text .npy tokens")
    parser.add_argument("--output_manifest", type=str, required=True,
                        help="Output TSV manifest path")
    parser.add_argument("--tokenizer_model", type=str, default=None,
                        help="SentencePiece model for text tokenization")
    parser.add_argument("--max_frames", type=int, default=250)
    parser.add_argument("--direction", type=str, default="en2sw",
                        choices=["en2sw", "sw2en"])
    args = parser.parse_args()

    if args.source == "fleurs":
        create_fleurs_manifest(
            en_token_dir=args.source_token_dir,
            sw_token_dir=args.target_token_dir,
            text_token_dir=args.text_token_dir,
            output_path=args.output_manifest,
            tokenizer_path=args.tokenizer_model,
            max_frames=args.max_frames,
        )
    elif args.source in ("kenspeech", "directory"):
        create_directory_manifest(
            source_dir=args.source_token_dir,
            target_dir=args.target_token_dir,
            text_dir=args.text_token_dir,
            output_path=args.output_manifest,
        )


if __name__ == "__main__":
    main()
