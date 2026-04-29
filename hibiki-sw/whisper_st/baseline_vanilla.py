"""Vanilla Whisper zero-shot ST baseline (no fine-tuning, no lexicon).

Generates predictions on the same test wavs as inference.py for fair comparison.
Uses Whisper's built-in `task="translate"` mode which does end-to-end Sw->En
speech translation in one decoder pass (no separate transcript step).

Usage:
    python whisper_st/baseline_vanilla.py \
        --base_model openai/whisper-small \
        --audio_dir /kaggle/working/fleurs_sw_ke_test/audio \
        --output_path /kaggle/working/preds_vanilla.jsonl

For a stronger baseline, use Whisper-large-v3 with device_map="auto":
    python whisper_st/baseline_vanilla.py \
        --base_model openai/whisper-large-v3 \
        --audio_dir /kaggle/working/fleurs_sw_ke_test/audio \
        --output_path /kaggle/working/preds_vanilla_largev3.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="openai/whisper-small")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device_map", type=str, default=None,
                        help='Set to "auto" to shard a large model across multiple GPUs.')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.base_model}...")
    processor = WhisperProcessor.from_pretrained(args.base_model, language="sw", task="translate")

    if args.device_map:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map=args.device_map,
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
    model.eval()

    # Force Whisper into Sw->En translation mode
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="sw", task="translate")

    wavs = sorted(Path(args.audio_dir).glob("*.wav"))
    if args.max_samples:
        wavs = wavs[: args.max_samples]
    print(f"Translating {len(wavs)} files (zero-shot Whisper Sw->En)...")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for i, wav_path in enumerate(wavs):
            audio, sr = sf.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import torchaudio
                wav_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
                audio = wav_t.squeeze(0).numpy()

            inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device, dtype=model.dtype)

            with torch.no_grad():
                generated = model.generate(
                    input_features=input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                )
            translation = processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

            out_f.write(json.dumps({"audio": wav_path.name, "translation": translation}, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(wavs)}  -> {translation[:80]}")

    print(f"\nDone! {len(wavs)} predictions -> {args.output_path}")


if __name__ == "__main__":
    main()
