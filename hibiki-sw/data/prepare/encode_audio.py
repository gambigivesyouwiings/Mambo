"""Encode audio datasets to Mimi codec tokens (.npy files).

Processes audio files through the frozen Mimi neural codec and saves the
discrete token representations as .npy arrays of shape (Q, T).

Supports:
    - Common Voice (en, sw) for Stage 2 audio pretraining
    - FLEURS (en_us, sw_ke) for evaluation
    - Custom audio directories

Usage:
    python data/prepare/encode_audio.py \
        --source common_voice \
        --lang en \
        --output_dir /path/to/audio_tokens/en \
        --num_codebooks 8 \
        --batch_size 8
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def load_mimi_codec(num_codebooks: int = 8, device: str = "cuda"):
    """Load the Mimi codec model."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from model.codec import MimiCodec

    codec = MimiCodec(num_codebooks=num_codebooks, device=device)
    return codec


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int = 24000) -> torch.Tensor:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def encode_common_voice(
    lang: str,
    output_dir: str,
    codec,
    max_samples: Optional[int] = None,
    max_duration_sec: float = 20.0,
):
    """Encode Common Voice dataset to Mimi tokens."""
    from datasets import load_dataset

    print(f"Loading Common Voice ({lang})...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_16_0",
        lang, split="train", trust_remote_code=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for i, sample in enumerate(tqdm(ds, desc=f"Encoding CV-{lang}")):
        if max_samples and count >= max_samples:
            break

        try:
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
            sr = audio["sampling_rate"]

            # Skip very long or very short audio
            duration = waveform.shape[1] / sr
            if duration < 1.0 or duration > max_duration_sec:
                continue

            # Resample to 24kHz
            waveform = resample_audio(waveform, sr, 24000)

            # Encode
            waveform = waveform.unsqueeze(0)  # (1, 1, samples)
            tokens = codec.encode(waveform)  # (1, Q, T)
            tokens_np = tokens[0].cpu().numpy().astype(np.int16)

            # Save
            out_path = os.path.join(output_dir, f"{lang}_{i:07d}.npy")
            np.save(out_path, tokens_np)
            count += 1

        except Exception as e:
            print(f"  Skipping sample {i}: {e}")
            continue

    print(f"Encoded {count} samples from Common Voice ({lang}) -> {output_dir}")
    return count


def encode_fleurs(
    lang: str,
    output_dir: str,
    codec,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """Encode FLEURS dataset to Mimi tokens."""
    from datasets import load_dataset

    print(f"Loading FLEURS ({lang}, {split})...")
    ds = load_dataset("google/fleurs", lang, split=split, trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for i, sample in enumerate(tqdm(ds, desc=f"Encoding FLEURS-{lang}")):
        if max_samples and count >= max_samples:
            break

        try:
            audio = sample["audio"]
            waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
            sr = audio["sampling_rate"]

            waveform = resample_audio(waveform, sr, 24000)
            waveform = waveform.unsqueeze(0)

            tokens = codec.encode(waveform)
            tokens_np = tokens[0].cpu().numpy().astype(np.int16)

            out_path = os.path.join(output_dir, f"fleurs_{lang}_{split}_{i:05d}.npy")
            np.save(out_path, tokens_np)
            count += 1

        except Exception as e:
            print(f"  Skipping sample {i}: {e}")
            continue

    print(f"Encoded {count} samples from FLEURS ({lang}, {split}) -> {output_dir}")
    return count


def encode_audio_dir(
    audio_dir: str,
    output_dir: str,
    codec,
    extensions: tuple = (".wav", ".flac", ".mp3", ".ogg"),
    max_duration_sec: float = 20.0,
):
    """Encode all audio files in a directory to Mimi tokens."""
    audio_dir = Path(audio_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = []
    for ext in extensions:
        files.extend(audio_dir.glob(f"**/*{ext}"))
    files.sort()

    print(f"Found {len(files)} audio files in {audio_dir}")
    count = 0

    for audio_path in tqdm(files, desc="Encoding audio"):
        try:
            waveform, sr = torchaudio.load(str(audio_path))

            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            duration = waveform.shape[1] / sr
            if duration < 0.5 or duration > max_duration_sec:
                continue

            waveform = resample_audio(waveform, sr, 24000)
            waveform = waveform.unsqueeze(0)

            tokens = codec.encode(waveform)
            tokens_np = tokens[0].cpu().numpy().astype(np.int16)

            stem = audio_path.stem
            out_path = os.path.join(output_dir, f"{stem}.npy")
            np.save(out_path, tokens_np)
            count += 1

        except Exception as e:
            print(f"  Skipping {audio_path.name}: {e}")
            continue

    print(f"Encoded {count} files -> {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Encode audio to Mimi tokens")
    parser.add_argument("--source", type=str, required=True,
                        choices=["common_voice", "fleurs", "directory"],
                        help="Audio source to encode")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language code (e.g. en, sw, en_us, sw_ke)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Audio directory (for --source directory)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_duration", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    codec = load_mimi_codec(args.num_codebooks, args.device)

    if args.source == "common_voice":
        encode_common_voice(
            args.lang, args.output_dir, codec,
            max_samples=args.max_samples,
            max_duration_sec=args.max_duration,
        )
    elif args.source == "fleurs":
        encode_fleurs(
            args.lang, args.output_dir, codec,
            split=args.split,
            max_samples=args.max_samples,
        )
    elif args.source == "directory":
        if not args.audio_dir:
            parser.error("--audio_dir is required for --source directory")
        encode_audio_dir(
            args.audio_dir, args.output_dir, codec,
            max_duration_sec=args.max_duration,
        )


if __name__ == "__main__":
    main()
