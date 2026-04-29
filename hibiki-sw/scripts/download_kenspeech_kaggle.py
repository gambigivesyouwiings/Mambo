"""
Download KenSpeech from HuggingFace and package it as a Kaggle dataset.

Run once locally:
    pip install datasets soundfile numpy tqdm huggingface_hub
    python hibiki-sw/scripts/download_kenspeech_kaggle.py

Then upload to Kaggle:
    kaggle datasets create -p kenspeech-sw

The dataset will be available at /kaggle/input/kenspeech-sw/ in notebooks.
"""

import io
import json
import os
import csv
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

from tqdm import tqdm


def decode_audio(raw_bytes: bytes):
    """Decode MP4/AAC audio bytes to mono float32 array using PyAV."""
    container = av.open(io.BytesIO(raw_bytes))
    sr = container.streams.audio[0].codec_context.sample_rate
    frames = []
    for frame in container.decode(audio=0):
        frames.append(frame.to_ndarray())
    # shape: (channels, samples) after concatenation
    array = np.concatenate(frames, axis=1).astype(np.float32)
    # Mix to mono if stereo
    if array.ndim == 2 and array.shape[0] > 1:
        array = array.mean(axis=0)
    else:
        array = array.squeeze()
    # Normalize if PCM int16 range
    if np.abs(array).max() > 1.0:
        array /= 32768.0
    return array, sr

# HuggingFace token — set via environment variable before running:
#   PowerShell: $env:HF_TOKEN="hf_..."
#   CMD:        set HF_TOKEN=hf_...
HF_TOKEN = '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'

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("HuggingFace login successful")
else:
    print("Warning: HF_TOKEN not set. Downloads may be slow or rate-limited.")
    print("  Set it with: $env:HF_TOKEN='hf_...' (PowerShell)")

# Output directory — will be uploaded as a Kaggle dataset
OUTPUT_DIR = Path("kenspeech-sw")
AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

DATASET_METADATA = {
    "title": "KenSpeech Swahili",
    "id": "victormugambi/kenspeech-sw",
    "licenses": [{"name": "CC-BY-4.0"}],
}

print("=" * 60)
print("KenSpeech Kaggle Packager")
print("=" * 60)
print(f"Output: {OUTPUT_DIR.resolve()}")
print("Loading KenSpeech (cached if already downloaded)...\n")

ds = load_dataset("Kencorpus/KenSpeech", split="train")
ds = ds.cast_column("audio", Audio(decode=False))  # get raw bytes, skip torchcodec
print(f"Loaded {len(ds)} samples\n")

# Save each sample as a WAV file + collect metadata
metadata_rows = []

# Resume from where we left off if WAVs already exist
existing = {f.stem for f in AUDIO_DIR.glob("*.wav")}
print(f"Resuming from {len(existing)} existing WAV files...\n" if existing else "")

skipped = 0
for i, row in enumerate(tqdm(ds, desc="Saving WAV files")):
    wav_filename = f"kenspeech_{i:05d}.wav"
    wav_path = AUDIO_DIR / wav_filename

    if wav_filename[:-4] in existing:
        # Already saved — verify it's valid, skip decoding if so
        try:
            info = sf.info(str(wav_path))
            metadata_rows.append({
                "path": f"audio/{wav_filename}",
                "transcript": row.get("transcript", ""),
                "speaker": row.get("speaker", ""),
                "gender": row.get("gender", ""),
                "duration": round(info.frames / info.samplerate, 3),
                "sample_rate": info.samplerate,
            })
            continue
        except Exception:
            # Corrupt file from a previous run — delete and re-decode
            wav_path.unlink(missing_ok=True)

    audio_data = row.get("audio", {})
    try:
        array, sr = decode_audio(audio_data["bytes"])
    except Exception as e:
        skipped += 1
        if skipped <= 10:
            print(f"\n  Skipping sample {i}: {e}")
        continue

    sf.write(str(wav_path), array, sr)
    metadata_rows.append({
        "path": f"audio/{wav_filename}",
        "transcript": row.get("transcript", ""),
        "speaker": row.get("speaker", ""),
        "gender": row.get("gender", ""),
        "duration": round(len(array) / sr, 3),
        "sample_rate": sr,
    })

if skipped:
    print(f"\nSkipped {skipped} corrupted samples out of {len(ds)}")

# Save metadata CSV
csv_path = OUTPUT_DIR / "metadata.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
    writer.writeheader()
    writer.writerows(metadata_rows)

# Save Kaggle dataset-metadata.json
with open(OUTPUT_DIR / "dataset-metadata.json", "w") as f:
    json.dump(DATASET_METADATA, f, indent=2)

total_size = sum(f.stat().st_size for f in AUDIO_DIR.glob("*.wav"))
print(f"\nSaved {len(metadata_rows)} WAV files ({total_size / 1e9:.2f} GB) -> {AUDIO_DIR}")
print(f"Saved metadata.csv -> {csv_path}")
print("\n" + "=" * 60)
print("Next step — upload to Kaggle:")
print("    kaggle datasets create -p kenspeech-sw")
print("=" * 60)
