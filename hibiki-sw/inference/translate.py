"""End-to-end speech translation inference pipeline.

Takes an English audio file and produces Swahili speech + text translation.

Usage:
    python inference/translate.py \
        --checkpoint /path/to/stage4/checkpoint_final.pt \
        --config configs/model_100m.yaml \
        --input audio.wav \
        --output_audio translated.wav \
        --output_text translated.txt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.hibiki_model import HibikiModel
from model.codec import MimiCodec


def load_model(config: dict, checkpoint_path: str, device: str = "cuda") -> HibikiModel:
    """Load trained Hibiki model from checkpoint."""
    model = HibikiModel.from_config(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))

    # Strip DDP prefix if present
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    model.load_state_dict(clean_state, strict=False)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters")
    return model


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


@torch.no_grad()
def translate(
    model: HibikiModel,
    codec: MimiCodec,
    source_audio: torch.Tensor,
    device: str = "cuda",
    temperature: float = 0.8,
    top_k_audio: int = 250,
    top_k_text: int = 50,
    voice_category: int = 3,  # "good"
    cfg_gamma: float = 3.0,
    max_len: int = 250,
) -> dict:
    """Translate source audio to target speech + text.

    Args:
        model: Trained HibikiModel
        codec: Mimi codec for encoding/decoding
        source_audio: (1, samples) waveform at 24kHz
        device: torch device
        temperature: audio sampling temperature
        top_k_audio: top-k for audio tokens
        top_k_text: top-k for text tokens
        voice_category: voice quality category (0-4)
        cfg_gamma: classifier-free guidance weight
        max_len: maximum output length in frames

    Returns:
        dict with translated_audio (waveform), translated_tokens (text ids),
        source_tokens, timing info
    """
    start_time = time.time()

    # Encode source audio to tokens
    source_audio = source_audio.unsqueeze(0).to(device)  # (1, 1, samples)
    source_tokens = codec.encode(source_audio)  # (1, Q, T)
    encode_time = time.time() - start_time

    print(f"Source: {source_audio.shape[-1] / 24000:.1f}s audio -> {source_tokens.shape[-1]} frames")

    # Voice category tensor
    vc = torch.tensor([voice_category], device=device, dtype=torch.long)

    # CFG: prepare "bad" category for guidance
    cfg_bad = torch.tensor([0], device=device, dtype=torch.long) if cfg_gamma > 1.0 else None

    # Generate translation
    gen_start = time.time()
    generated_audio_tokens, generated_text_tokens = model.generate(
        source_audio_tokens=source_tokens,
        max_len=max_len,
        temperature=temperature,
        top_k_audio=top_k_audio,
        top_k_text=top_k_text,
        voice_category=vc,
        cfg_gamma=cfg_gamma,
        cfg_bad_category=cfg_bad,
    )
    gen_time = time.time() - gen_start

    # Decode generated audio tokens to waveform
    decode_start = time.time()
    translated_waveform = codec.decode(generated_audio_tokens)  # (1, 1, samples)
    decode_time = time.time() - decode_start

    total_time = time.time() - start_time
    output_frames = generated_audio_tokens.shape[-1]
    output_duration = output_frames / 12.5

    print(f"Generated: {output_frames} frames ({output_duration:.1f}s)")
    print(f"Timing: encode={encode_time:.2f}s, generate={gen_time:.2f}s, "
          f"decode={decode_time:.2f}s, total={total_time:.2f}s")
    print(f"RTF: {total_time / (source_audio.shape[-1] / 24000):.2f}x")

    return {
        "translated_audio": translated_waveform[0].cpu(),  # (1, samples)
        "translated_text_ids": generated_text_tokens[0].cpu(),  # (T,)
        "source_tokens": source_tokens[0].cpu(),
        "target_tokens": generated_audio_tokens[0].cpu(),
        "encode_time": encode_time,
        "generate_time": gen_time,
        "decode_time": decode_time,
        "total_time": total_time,
    }


def decode_text(text_ids: torch.Tensor, tokenizer_path: str) -> str:
    """Decode text token IDs back to string."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # Filter out special tokens (pad=0, bos=1, eos=2, epad=3)
    ids = text_ids.tolist()
    filtered = [tid for tid in ids if tid >= 4]

    # Offset by -4 since model adds +4 for special tokens
    adjusted = [tid - 4 for tid in filtered if (tid - 4) >= 0 and (tid - 4) < sp.get_piece_size()]

    return sp.decode(adjusted)


def main():
    parser = argparse.ArgumentParser(description="Hibiki-Sw Speech Translation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to model config YAML")
    parser.add_argument("--input", type=str, required=True,
                        help="Input audio file path")
    parser.add_argument("--output_audio", type=str, default="translated.wav",
                        help="Output audio file path")
    parser.add_argument("--output_text", type=str, default=None,
                        help="Output text file path")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="SentencePiece tokenizer model path")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k_audio", type=int, default=250)
    parser.add_argument("--top_k_text", type=int, default=50)
    parser.add_argument("--voice_category", type=int, default=3,
                        help="Voice quality (0=very_bad, 4=very_good)")
    parser.add_argument("--cfg_gamma", type=float, default=3.0,
                        help="Classifier-free guidance weight (1.0=disabled)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    print("Loading model...")
    model = load_model(config, args.checkpoint, args.device)
    codec = MimiCodec(
        num_codebooks=config["model"]["codec"]["num_codebooks"],
        device=args.device,
    )

    # Load input audio
    print(f"Loading audio: {args.input}")
    source_audio = load_audio(args.input, target_sr=24000)
    print(f"Input: {source_audio.shape[-1] / 24000:.1f}s ({source_audio.shape[-1]} samples)")

    # Translate
    result = translate(
        model, codec, source_audio,
        device=args.device,
        temperature=args.temperature,
        top_k_audio=args.top_k_audio,
        top_k_text=args.top_k_text,
        voice_category=args.voice_category,
        cfg_gamma=args.cfg_gamma,
        max_len=config["model"]["temporal"]["max_seq_len"],
    )

    # Save audio
    torchaudio.save(args.output_audio, result["translated_audio"], 24000)
    print(f"Saved translated audio: {args.output_audio}")

    # Decode and save text
    if args.tokenizer or args.output_text:
        tokenizer_path = args.tokenizer or config.get("tokenizer", {}).get("model_path")
        if tokenizer_path:
            text = decode_text(result["translated_text_ids"], tokenizer_path)
            print(f"Translated text: {text}")
            if args.output_text:
                with open(args.output_text, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved text: {args.output_text}")


if __name__ == "__main__":
    main()
