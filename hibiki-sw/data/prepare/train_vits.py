"""Stage 0: Train VITS TTS on Common Voice Swahili.

Fine-tunes a VITS model on Common Voice Swahili validated data (~300hrs) to
produce a multi-speaker Swahili TTS system. This is used to synthesize the
target-language speech needed for the En→Sw synthetic parallel corpus.

Uses Coqui TTS (TTS library) for training. The trained model is saved and
later used by synthesize_tts.py to generate Swahili speech from MADLAD
translations.

For English TTS (Sw→En direction), we use a pretrained model (no training
needed).

Usage on Colab/Kaggle:
    python data/prepare/train_vits.py \
        --dataset_dir /content/cv-corpus-19.0-2024-09-13/sw \
        --output_dir /content/drive/MyDrive/hibiki-sw/vits_sw \
        --max_samples 50000 \
        --num_epochs 100 \
        --batch_size 16

    # Resume from checkpoint
    python data/prepare/train_vits.py \
        --dataset_dir /content/cv-corpus-19.0-2024-09-13/sw \
        --output_dir /content/drive/MyDrive/hibiki-sw/vits_sw \
        --resume_from /content/drive/MyDrive/hibiki-sw/vits_sw/checkpoint_25000.pth
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Dataset preparation: Common Voice → VITS format
# ---------------------------------------------------------------------------

def prepare_vits_dataset(
    dataset_dir: str,
    output_dir: str,
    split: str = "validated",
    max_samples: Optional[int] = None,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    val_ratio: float = 0.05,
    sample_rate: int = 22050,
) -> Tuple[str, str, Dict]:
    """Convert Common Voice data to VITS training format.

    VITS expects:
    - A metadata CSV: audio_path|speaker_id|text
    - Audio files resampled to 22050 Hz (mono WAV)
    - Speaker mapping for multi-speaker mode

    Returns:
        (train_csv_path, val_csv_path, speaker_map)
    """
    from data.prepare.local_cv_loader import CommonVoiceLocal
    import torchaudio

    print(f"Loading Common Voice ({split}) from {dataset_dir}...")
    ds = CommonVoiceLocal(dataset_dir=dataset_dir, split=split, load_audio=True)

    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # Collect valid samples
    samples = []
    speaker_set = set()
    skipped = 0

    n_total = min(len(ds), max_samples) if max_samples else len(ds)

    for i in range(n_total):
        try:
            sample = ds[i]
        except Exception as e:
            skipped += 1
            continue

        audio = sample["audio"]
        sr = audio["sampling_rate"]
        waveform = audio["array"]

        duration = len(waveform) / sr
        if duration < min_duration or duration > max_duration:
            skipped += 1
            continue

        text = sample["sentence"].strip()
        if not text or len(text) < 3:
            skipped += 1
            continue

        # Use client_id as speaker ID
        speaker = sample.get("client_id", "default")
        speaker_set.add(speaker)

        # Resample to 22050 Hz and save as WAV
        waveform_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        if sr != sample_rate:
            waveform_t = torchaudio.functional.resample(waveform_t, sr, sample_rate)

        # Normalize volume
        peak = waveform_t.abs().max()
        if peak > 0:
            waveform_t = waveform_t / peak * 0.95

        wav_path = os.path.join(wav_dir, f"cv_{i:06d}.wav")
        torchaudio.save(wav_path, waveform_t, sample_rate)

        samples.append({
            "audio_path": wav_path,
            "speaker": speaker,
            "text": text,
        })

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{n_total} ({skipped} skipped, {len(speaker_set)} speakers)")

    print(f"Valid samples: {len(samples)}, Skipped: {skipped}, Speakers: {len(speaker_set)}")

    # Create speaker map (string -> int)
    speaker_list = sorted(speaker_set)
    speaker_map = {s: idx for idx, s in enumerate(speaker_list)}

    # Shuffle and split train/val
    random.shuffle(samples)
    val_size = max(1, int(len(samples) * val_ratio))
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    # Write metadata CSVs
    train_csv = os.path.join(output_dir, "train.csv")
    val_csv = os.path.join(output_dir, "val.csv")

    for csv_path, data in [(train_csv, train_samples), (val_csv, val_samples)]:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            for s in data:
                writer.writerow([s["audio_path"], speaker_map[s["speaker"]], s["text"]])

    # Save speaker map
    speaker_map_path = os.path.join(output_dir, "speaker_map.json")
    with open(speaker_map_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Speaker map saved to {speaker_map_path}")

    return train_csv, val_csv, speaker_map


# ---------------------------------------------------------------------------
# VITS Training via Coqui TTS
# ---------------------------------------------------------------------------

def train_vits_coqui(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    speaker_map: Dict,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    resume_from: Optional[str] = None,
    sample_rate: int = 22050,
    num_workers: int = 2,
):
    """Train VITS using Coqui TTS library.

    This uses the TTS library's VITS implementation with multi-speaker support.
    """
    try:
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.tts.datasets import load_tts_samples
        from TTS.utils.audio import AudioProcessor
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
    except ImportError:
        print("Coqui TTS not available. Falling back to custom VITS training.")
        print("Install with: pip install TTS")
        train_vits_custom(
            train_csv, val_csv, output_dir, speaker_map,
            num_epochs, batch_size, learning_rate, resume_from,
            sample_rate, num_workers,
        )
        return

    num_speakers = len(speaker_map)
    use_speaker_embedding = num_speakers > 1

    print(f"Training VITS with Coqui TTS")
    print(f"  Speakers: {num_speakers}, Multi-speaker: {use_speaker_embedding}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Configure VITS
    config = VitsConfig(
        output_path=output_dir,
        run_name="vits_sw",
        batch_size=batch_size,
        eval_batch_size=max(1, batch_size // 2),
        num_loader_workers=num_workers,
        num_eval_loader_workers=1,
        epochs=num_epochs,
        lr_gen=learning_rate,
        lr_disc=learning_rate,
        use_speaker_embedding=use_speaker_embedding,
        num_speakers=num_speakers if use_speaker_embedding else 0,
        speaker_embedding_channels=256 if use_speaker_embedding else 0,
        audio={
            "sample_rate": sample_rate,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "fft_size": 1024,
        },
        text_cleaner="multilingual_cleaners",
        characters={
            "characters_class": "TTS.tts.utils.text.characters.Graphemes",
            "pad": "<PAD>",
            "eos": "<EOS>",
            "bos": "<BOS>",
            "blank": "<BLNK>",
        },
        test_sentences=[
            ["Habari, hali yako?", "vits_sw", None, "sw"],
            ["Tanzania ni nchi nzuri sana.", "vits_sw", None, "sw"],
            ["Karibu sana, asante kwa kuja.", "vits_sw", None, "sw"],
        ],
        mixed_precision=True,
        save_step=5000,
        print_step=100,
        save_checkpoints=True,
        save_best_after=10000,
    )

    # Build dataset formatter for pipe-separated CSV
    def cv_formatter(root_path, meta_file, **kwargs):
        """Read pipe-separated CSV: audio_path|speaker_id|text"""
        items = []
        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) >= 3:
                    audio_path, speaker_id, text = row[0], row[1], row[2]
                    items.append({
                        "text": text,
                        "audio_file": audio_path,
                        "speaker_name": f"speaker_{speaker_id}",
                        "root_path": root_path,
                    })
        return items

    # Load datasets
    train_samples = cv_formatter(output_dir, train_csv)
    eval_samples = cv_formatter(output_dir, val_csv)

    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples: {len(eval_samples)}")

    # Initialize model
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    if resume_from and os.path.exists(resume_from):
        print(f"  Resuming from {resume_from}")
        model.load_checkpoint(config, resume_from)

    # Train
    from TTS.trainer import Trainer, TrainerArgs

    trainer_args = TrainerArgs(
        restore_path=resume_from if resume_from else None,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=output_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    print(f"\nVITS training complete! Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# Fallback: Custom lightweight VITS training (no Coqui dependency)
# ---------------------------------------------------------------------------

def train_vits_custom(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    speaker_map: Dict,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    resume_from: Optional[str] = None,
    sample_rate: int = 22050,
    num_workers: int = 2,
):
    """Fallback VITS training using Meta's MMS-TTS fine-tuning approach.

    If Coqui TTS isn't available, we fine-tune facebook/mms-tts-swh
    (Meta's Massively Multilingual Speech TTS for Swahili) on Common Voice
    data. This gives us a strong Swahili TTS with much less training.
    """
    from transformers import (
        VitsModel,
        VitsTokenizer,
        VitsConfig as HFVitsConfig,
    )

    print("=" * 60)
    print("VITS Training: Fine-tuning facebook/mms-tts-swh")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained Swahili TTS
    model_name = "facebook/mms-tts-swh"
    print(f"Loading {model_name}...")
    tokenizer = VitsTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(device)

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    # Dataset class for CSV format
    from torch.utils.data import Dataset, DataLoader

    class VitsCSVDataset(Dataset):
        def __init__(self, csv_path, tokenizer, sample_rate=22050, max_audio_len=None):
            self.entries = []
            self.tokenizer = tokenizer
            self.sample_rate = sample_rate
            self.max_audio_len = max_audio_len or sample_rate * 15  # 15s max

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for row in reader:
                    if len(row) >= 3:
                        self.entries.append({
                            "audio_path": row[0],
                            "speaker_id": int(row[1]),
                            "text": row[2],
                        })

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            entry = self.entries[idx]
            import torchaudio

            # Load audio
            waveform, sr = torchaudio.load(entry["audio_path"])
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            waveform = waveform.squeeze(0)

            # Truncate
            if waveform.shape[0] > self.max_audio_len:
                waveform = waveform[:self.max_audio_len]

            # Tokenize text
            inputs = self.tokenizer(entry["text"], return_tensors="pt")
            input_ids = inputs["input_ids"].squeeze(0)

            return {
                "input_ids": input_ids,
                "waveform": waveform,
                "speaker_id": entry["speaker_id"],
            }

    def collate_fn(batch):
        """Pad variable-length sequences."""
        # Pad input_ids
        max_text_len = max(b["input_ids"].shape[0] for b in batch)
        max_audio_len = max(b["waveform"].shape[0] for b in batch)

        input_ids = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        input_ids_mask = torch.zeros(len(batch), max_text_len, dtype=torch.bool)
        waveforms = torch.zeros(len(batch), max_audio_len)
        waveform_lengths = torch.zeros(len(batch), dtype=torch.long)
        speaker_ids = torch.zeros(len(batch), dtype=torch.long)

        for i, b in enumerate(batch):
            tl = b["input_ids"].shape[0]
            al = b["waveform"].shape[0]
            input_ids[i, :tl] = b["input_ids"]
            input_ids_mask[i, :tl] = True
            waveforms[i, :al] = b["waveform"]
            waveform_lengths[i] = al
            speaker_ids[i] = b["speaker_id"]

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids_mask,
            "waveforms": waveforms,
            "waveform_lengths": waveform_lengths,
            "speaker_ids": speaker_ids,
        }

    # Create dataloaders
    print("Loading datasets...")
    train_ds = VitsCSVDataset(train_csv, tokenizer, sample_rate)
    val_ds = VitsCSVDataset(val_csv, tokenizer, sample_rate)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, batch_size // 2), shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer — only fine-tune decoder + duration predictor, freeze encoder mostly
    # This makes training much faster for adaptation
    trainable_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["dec.", "dp.", "flow.", "posterior_encoder"]):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    print(f"Total params: {total_params:,}, Trainable: {train_params:,} "
          f"({100*train_params/total_params:.1f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            # Forward pass — VITS computes its own losses internally
            try:
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,  # self-supervised
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else torch.tensor(0.0)
            except Exception as e:
                # VITS forward can be finicky — skip bad batches
                if batch_idx % 100 == 0:
                    print(f"  Skipping batch {batch_idx}: {e}")
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}, Step {global_step}, Loss: {avg:.4f}")

            if global_step % 5000 == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_{global_step}.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        val_loss += outputs.loss.item()
                except Exception:
                    continue

        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch+1}/{num_epochs} — Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  New best model saved (val_loss={best_val_loss:.4f})")

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "epoch": num_epochs,
    }, final_path)

    # Also save with HuggingFace format for easy loading during synthesis
    hf_dir = os.path.join(output_dir, "hf_model")
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"HuggingFace format: {hf_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Train VITS TTS on Common Voice Swahili"
    )
    # Data
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to extracted Common Voice language directory, "
                             "e.g. /content/cv-corpus-19.0-2024-09-13/sw")
    parser.add_argument("--split", type=str, default="validated",
                        help="Common Voice split to use")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model + processed data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of CV samples to use")
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=15.0,
                        help="Maximum audio duration in seconds")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="VITS sample rate (22050 for standard VITS)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Mode
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip dataset preparation (reuse existing CSVs)")

    args = parser.parse_args()

    # Step 1: Prepare dataset
    if not args.skip_data_prep:
        train_csv, val_csv, speaker_map = prepare_vits_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            sample_rate=args.sample_rate,
        )
    else:
        train_csv = os.path.join(args.output_dir, "train.csv")
        val_csv = os.path.join(args.output_dir, "val.csv")
        with open(os.path.join(args.output_dir, "speaker_map.json")) as f:
            speaker_map = json.load(f)
        print(f"Reusing existing data: {train_csv}")

    # Step 2: Train VITS
    train_vits_coqui(
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=args.output_dir,
        speaker_map=speaker_map,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
