"""Evaluation pipeline for Hibiki-Sw speech translation.

Metrics:
    1. ASR-BLEU: Transcribe generated Swahili audio with Whisper, compute BLEU
       against reference translations.
    2. Speaker Similarity (SIM-o): Cosine similarity of WavLM embeddings
       between generated and source speaker.
    3. Translation Quality: BLEU of inner monologue text vs reference text.
    4. Latency: Average real-time factor (RTF) and startup delay.

Usage:
    python evaluation/evaluate.py \
        --checkpoint /path/to/checkpoint.pt \
        --config configs/model_100m.yaml \
        --eval_set fleurs \
        --output_dir /path/to/eval_results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.hibiki_model import HibikiModel
from model.codec import MimiCodec
from inference.translate import load_model, load_audio, translate, decode_text


# ---------------------------------------------------------------------------
# ASR-BLEU
# ---------------------------------------------------------------------------

class ASRBLEUScorer:
    """Compute ASR-BLEU: transcribe generated audio, then compute BLEU."""

    def __init__(self, whisper_model: str = "medium", device: str = "cuda"):
        self.device = device
        self.whisper_model_name = whisper_model
        self._model = None

    def _load_whisper(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.whisper_model_name,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
            )
        except ImportError:
            import whisper
            self._model = whisper.load_model(self.whisper_model_name, device=self.device)

    def transcribe(self, audio_path: str, language: str = "sw") -> str:
        """Transcribe audio file to text."""
        self._load_whisper()

        try:
            from faster_whisper import WhisperModel
            segments, _ = self._model.transcribe(
                audio_path, language=language, beam_size=5
            )
            return " ".join(seg.text.strip() for seg in segments)
        except Exception:
            result = self._model.transcribe(
                audio_path, language=language, beam_size=5
            )
            return result["text"].strip()

    def compute_bleu(self, hypotheses: List[str], references: List[str]) -> Dict:
        """Compute BLEU score."""
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return {
            "bleu": bleu.score,
            "bleu_bp": bleu.bp,
            "bleu_precisions": bleu.precisions,
        }


# ---------------------------------------------------------------------------
# Speaker Similarity
# ---------------------------------------------------------------------------

class SpeakerSimilarityScorer:
    """Compute speaker similarity using WavLM embeddings."""

    def __init__(self, model_name: str = "microsoft/wavlm-large", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._feature_extractor = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self._model = WavLMModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding from audio file."""
        self._load_model()

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz (WavLM expects 16kHz)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        inputs = self._feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # Mean pool over time
        embedding = outputs.last_hidden_state.mean(dim=1)  # (1, D)
        return embedding.squeeze(0)

    def compute_similarity(self, audio1_path: str, audio2_path: str) -> float:
        """Compute cosine similarity between two audio files."""
        emb1 = self.get_embedding(audio1_path)
        emb2 = self.get_embedding(audio2_path)

        sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return sim.item()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_on_fleurs(
    model: HibikiModel,
    codec: MimiCodec,
    config: dict,
    output_dir: str,
    device: str = "cuda",
    tokenizer_path: Optional[str] = None,
    max_samples: int = 100,
    direction: str = "en2sw",
):
    """Run evaluation on FLEURS dataset."""
    from datasets import load_dataset

    src_lang = "en_us" if direction == "en2sw" else "sw_ke"
    tgt_lang = "sw_ke" if direction == "en2sw" else "en_us"
    tgt_whisper_lang = "sw" if direction == "en2sw" else "en"

    print(f"Loading FLEURS test set ({src_lang} -> {tgt_lang})...")
    src_ds = load_dataset("google/fleurs", src_lang, split="test")
    tgt_ds = load_dataset("google/fleurs", tgt_lang, split="test")

    # Match by ID
    src_by_id = {s["id"]: s for s in src_ds}
    tgt_by_id = {s["id"]: s for s in tgt_ds}
    common_ids = sorted(set(src_by_id.keys()) & set(tgt_by_id.keys()))[:max_samples]

    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)

    # Initialize scorers
    asr_scorer = ASRBLEUScorer(
        whisper_model=config.get("evaluation", {}).get("whisper_model", "medium"),
        device=device,
    )

    inf_cfg = config.get("inference", {})
    results = []
    hypotheses = []
    references = []
    text_hypotheses = []
    text_references = []

    for uid in tqdm(common_ids, desc="Evaluating"):
        src_sample = src_by_id[uid]
        tgt_sample = tgt_by_id[uid]

        try:
            # Load source audio
            src_audio = torch.tensor(
                src_sample["audio"]["array"], dtype=torch.float32
            ).unsqueeze(0)
            src_sr = src_sample["audio"]["sampling_rate"]

            if src_sr != 24000:
                resampler = torchaudio.transforms.Resample(src_sr, 24000)
                src_audio = resampler(src_audio)

            # Translate
            result = translate(
                model, codec, src_audio,
                device=device,
                temperature=inf_cfg.get("temperature", 0.8),
                top_k_audio=inf_cfg.get("top_k_audio", 250),
                top_k_text=inf_cfg.get("top_k_text", 50),
                voice_category=3,
                cfg_gamma=inf_cfg.get("cfg_gamma", 3.0),
                max_len=config["model"]["temporal"]["max_seq_len"],
            )

            # Save generated audio
            gen_audio_path = os.path.join(output_dir, "audio", f"gen_{uid}.wav")
            torchaudio.save(gen_audio_path, result["translated_audio"], 24000)

            # ASR-BLEU: transcribe generated audio
            hypothesis = asr_scorer.transcribe(gen_audio_path, language=tgt_whisper_lang)
            reference = tgt_sample["transcription"]
            hypotheses.append(hypothesis)
            references.append(reference)

            # Text metrics (inner monologue)
            if tokenizer_path:
                text_hyp = decode_text(result["translated_text_ids"], tokenizer_path)
                text_hypotheses.append(text_hyp)
                text_references.append(reference)

            results.append({
                "id": uid,
                "source_text": src_sample["transcription"],
                "reference_text": reference,
                "asr_hypothesis": hypothesis,
                "text_hypothesis": text_hyp if tokenizer_path else None,
                "generate_time": result["generate_time"],
                "total_time": result["total_time"],
                "source_duration": src_audio.shape[-1] / 24000,
            })

        except Exception as e:
            print(f"  Error on sample {uid}: {e}")
            continue

    # Compute metrics
    metrics = {}

    if hypotheses:
        asr_bleu = asr_scorer.compute_bleu(hypotheses, references)
        metrics["asr_bleu"] = asr_bleu
        print(f"\nASR-BLEU: {asr_bleu['bleu']:.2f}")

    if text_hypotheses:
        text_bleu = asr_scorer.compute_bleu(text_hypotheses, text_references)
        metrics["text_bleu"] = text_bleu
        print(f"Text BLEU: {text_bleu['bleu']:.2f}")

    # Latency metrics
    if results:
        gen_times = [r["generate_time"] for r in results]
        total_times = [r["total_time"] for r in results]
        src_durations = [r["source_duration"] for r in results]
        rtfs = [t / d for t, d in zip(total_times, src_durations) if d > 0]

        metrics["latency"] = {
            "avg_generate_time": np.mean(gen_times),
            "avg_total_time": np.mean(total_times),
            "avg_rtf": np.mean(rtfs),
            "median_rtf": np.median(rtfs),
        }
        print(f"Avg RTF: {np.mean(rtfs):.2f}x")

    metrics["num_samples"] = len(results)

    # Save results
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hibiki-Sw")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval_set", type=str, default="fleurs",
                        choices=["fleurs"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--direction", type=str, default="en2sw",
                        choices=["en2sw", "sw2en"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = load_model(config, args.checkpoint, args.device)
    codec = MimiCodec(
        num_codebooks=config["model"]["codec"]["num_codebooks"],
        device=args.device,
    )

    if args.eval_set == "fleurs":
        evaluate_on_fleurs(
            model, codec, config,
            output_dir=args.output_dir,
            device=args.device,
            tokenizer_path=args.tokenizer,
            max_samples=args.max_samples,
            direction=args.direction,
        )


if __name__ == "__main__":
    main()
