"""Inference for transcript-prompted Whisper with optional lexicon augmentation.

At inference time:
  1. Encode audio
  2. CTC-greedy decode the encoder hidden states -> Sw transcript hypothesis
  3. (Optional) Look up transcript words/bigrams in the bilingual lexicon
  4. Build the decoder prompt:
       <|sot|><|sw|><|transcribe|><|notimestamps|>{transcript}{lexicon_hints}<|en|><|translate|><|notimestamps|>
  5. Decoder generates En translation autoregressively

Usage:
    python whisper_st/inference.py \
        --model_dir /kaggle/working/whisper_st_sw2en/final \
        --audio_dir /kaggle/input/fleurs-sw-test/audio \
        --output_path /kaggle/working/predictions_sw2en.jsonl \
        --lexicon_path /kaggle/working/lexicon.jsonl

Outputs JSONL with one entry per audio file:
    {"audio": "...", "transcript_ctc": "...", "lexicon_hits": [...], "translation": "..."}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from whisper_st.model import TranscriptPromptedWhisper


WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def load_lexicon(path: str) -> Dict[str, str]:
    """Load JSONL lexicon into {sw: en} dict (lowercased keys)."""
    lex: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                lex[e["sw"].lower()] = e["en"]
            except Exception:
                continue
    print(f"Loaded {len(lex)} lexicon entries from {path}")
    return lex


def lookup_lexicon(transcript: str, lexicon: Dict[str, str], max_hits: int = 6) -> List[Dict[str, str]]:
    """Find Sw words in transcript that have lexicon entries.

    Tries unigrams first, then bigrams. Returns up to max_hits unique matches
    in transcript order.
    """
    hits = []
    seen = set()
    words = WORD_RE.findall(transcript.lower())
    # Bigrams
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg in lexicon and bg not in seen:
            hits.append({"sw": bg, "en": lexicon[bg]})
            seen.add(bg)
    # Unigrams
    for w in words:
        if w in lexicon and w not in seen:
            hits.append({"sw": w, "en": lexicon[w]})
            seen.add(w)
    return hits[:max_hits]


def format_lexicon_hint(hits: List[Dict[str, str]]) -> str:
    """Format lexicon matches as a natural-language-ish hint string.

    Example: " (Nairobi=Nairobi, mji mkuu=capital city)"
    Empty string if no hits. We surround with parentheses so the decoder
    sees them as a parenthetical aside to the transcript.
    """
    if not hits:
        return ""
    pairs = ", ".join(f"{h['sw']}={h['en']}" for h in hits)
    return f" ({pairs})"


@torch.no_grad()
def translate_one(
    audio: np.ndarray,
    model: TranscriptPromptedWhisper,
    processor: WhisperProcessor,
    lexicon: Optional[Dict[str, str]] = None,
    max_new_tokens: int = 200,
    device: str = "cuda",
    forced_transcript: Optional[str] = None,
) -> Dict:
    """Run the full transcript-prompted ST pipeline on a single audio array.

    Transcription uses the Whisper decoder in sw+transcribe mode (more reliable
    than the CTC head at low data regimes). Translation then uses the fine-tuned
    decoder prompted with the recovered transcript.

    If `forced_transcript` is supplied, Pass 1 is skipped and that string is
    used as the prompt for Pass 2 directly. Used to isolate the translation
    head's quality from the (possibly broken) autoregressive transcription pass.
    """
    tokenizer = processor.tokenizer

    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    sw = tokenizer.convert_tokens_to_ids("<|sw|>")
    en = tokenizer.convert_tokens_to_ids("<|en|>")
    transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    translate = tokenizer.convert_tokens_to_ids("<|translate|>")
    notim = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eot_id = tokenizer.eos_token_id

    encoder_outputs = model.model.encoder(input_features=feats, return_dict=True)
    encoder_hidden = encoder_outputs.last_hidden_state

    def greedy_decode(prompt, stop_extra=()):
        ids = prompt[:]
        for _ in range(max_new_tokens):
            dec_out = model.model.decoder(
                input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                encoder_hidden_states=encoder_hidden,
                return_dict=True,
            )
            nxt = model.proj_out(dec_out.last_hidden_state[:, -1, :]).argmax(dim=-1).item()
            ids.append(nxt)
            if nxt == eot_id or nxt in stop_extra:
                break
        return ids

    # 1) Transcribe — use forced transcript if provided, else autoregressive Pass 1
    if forced_transcript is not None:
        transcript = forced_transcript.strip()
    else:
        trans_ids = greedy_decode([sot, sw, transcribe, notim], stop_extra=(en,))
        transcript_token_ids = [t for t in trans_ids[4:] if t not in (eot_id, en)]
        transcript = tokenizer.decode(transcript_token_ids, skip_special_tokens=True).strip()

    # 2) Lexicon lookup (optional)
    lexicon_hits = []
    hint = ""
    if lexicon:
        lexicon_hits = lookup_lexicon(transcript, lexicon)
        hint = format_lexicon_hint(lexicon_hits)

    # 3) Build translation prompt and decode
    prompt_ids = tokenizer(transcript + hint, add_special_tokens=False).input_ids
    translation_prompt = [sot, sw, transcribe, notim] + prompt_ids + [en, translate, notim]
    gen_ids = greedy_decode(translation_prompt)
    new_tokens = [t for t in gen_ids[len(translation_prompt):] if t != eot_id]
    translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "transcript_ctc": transcript,
        "lexicon_hits": lexicon_hits,
        "translation": translation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Trained TranscriptPromptedWhisper directory")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory of WAV files to translate")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lexicon_path", type=str, default=None,
                        help="Optional JSONL lexicon for prompt augmentation")
    parser.add_argument("--reference_transcripts", type=str, default=None,
                        help="Diagnostic: JSONL with {audio, reference_sw} entries. When set, "
                             "skips Pass 1 (autoregressive transcription) and uses the reference "
                             "Sw transcript as the Pass 2 prompt directly. Isolates translation "
                             "quality from transcription quality.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir} on {device}...")
    processor = WhisperProcessor.from_pretrained(args.model_dir, language="sw", task="transcribe")
    model = TranscriptPromptedWhisper.from_pretrained(args.model_dir).to(device)
    model.eval()

    lexicon = load_lexicon(args.lexicon_path) if args.lexicon_path else None

    ref_transcripts: Optional[Dict[str, str]] = None
    if args.reference_transcripts:
        ref_transcripts = {}
        with open(args.reference_transcripts, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if "audio" in e and "reference_sw" in e:
                        ref_transcripts[e["audio"]] = e["reference_sw"]
                except Exception:
                    continue
        print(f"Loaded {len(ref_transcripts)} reference transcripts (Pass 1 will be skipped)")

    wavs = sorted(Path(args.audio_dir).glob("*.wav"))
    if args.max_samples:
        wavs = wavs[: args.max_samples]
    print(f"Translating {len(wavs)} files...")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for i, wav_path in enumerate(wavs):
            audio, sr = sf.read(str(wav_path))
            if sr != 16000:
                # Quick resample (use torchaudio for proper resampling in production)
                import torchaudio
                wav_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
                audio = wav_t.squeeze(0).numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            forced = ref_transcripts.get(wav_path.name) if ref_transcripts else None
            if ref_transcripts and forced is None:
                # Skip files we don't have a reference transcript for, in diag mode
                continue

            try:
                result = translate_one(
                    audio, model, processor, lexicon=lexicon,
                    max_new_tokens=args.max_new_tokens, device=device,
                    forced_transcript=forced,
                )
            except Exception as e:
                print(f"  Error on {wav_path.name}: {e}")
                continue

            entry = {"audio": wav_path.name, **result}
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(wavs)}  "
                      f"[sw] {result['transcript_ctc'][:60]} -> [en] {result['translation'][:60]}")

    print(f"\nDone! Predictions -> {args.output_path}")


if __name__ == "__main__":
    main()
