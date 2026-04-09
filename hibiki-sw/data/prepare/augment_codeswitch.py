"""Sheng / code-switch augmentation via LLM APIs (Grok, Claude, or OpenAI).

Generates realistic Swahili-English code-switched (Sheng) text variants from
clean translations, for training the Hibiki model on real-world mixed-language
speech patterns common in East African conversation.

Sheng patterns include:
    - Lexical borrowing: "ni-drive" "ana-text" "wame-cancel"
    - Intra-sentential switching: "Sasa bro, uko place gani?"
    - Tag switching: "Ni sawa, right?" "It's fine, sawa?"
    - Borrowed English with Swahili morphology: "ku-connect" "a-li-enjoy"

Pipeline integration:
    clean translations (MADLAD) → [this script] → Sheng variants → VITS TTS → training

Supports three LLM backends:
    - Grok (xAI): Best for Sheng, trained on X/Twitter data
    - Claude (Anthropic): High quality, reliable
    - OpenAI: GPT-4 fallback

Usage:
    # Generate Sheng variants from existing translations
    python data/prepare/augment_codeswitch.py \
        --input_dir /content/drive/MyDrive/hibiki-sw/translations/en2sw \
        --output_dir /content/drive/MyDrive/hibiki-sw/translations/en2sw_sheng \
        --backend grok \
        --api_key $GROK_API_KEY \
        --max_samples 10000 \
        --batch_size 20

    # Generate standalone Sheng text corpus (for tokenizer training)
    python data/prepare/augment_codeswitch.py \
        --mode generate_corpus \
        --output_dir /content/drive/MyDrive/hibiki-sw/sheng_corpus \
        --backend claude \
        --api_key $ANTHROPIC_API_KEY \
        --num_samples 50000

    # Rule-based augmentation (no API needed, free but lower quality)
    python data/prepare/augment_codeswitch.py \
        --input_dir /content/drive/MyDrive/hibiki-sw/translations/en2sw \
        --output_dir /content/drive/MyDrive/hibiki-sw/translations/en2sw_sheng \
        --backend rules \
        --max_samples 50000
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Sheng lexicon and rule-based patterns (no API needed)
# ---------------------------------------------------------------------------

# Common Sheng substitutions: English → Sheng/mixed form
SHENG_LEXICON = {
    # Greetings & social
    "hello": ["sasa", "niaje", "mambo"],
    "hi": ["sasa", "niaje", "yo"],
    "how are you": ["vipi", "niaje", "uko aje"],
    "fine": ["poa", "safi", "fiti"],
    "good": ["poa", "safi", "fiti", "best"],
    "friend": ["bro", "msee", "manze", "dude"],
    "man": ["msee", "jamaa", "buda"],
    "woman": ["dame", "madam", "manzi"],
    "girl": ["manzi", "dem", "shorty"],
    "boy": ["kijanaa", "boi", "chali"],
    "money": ["doh", "munde", "chapaa", "cash"],
    "food": ["dishi", "chakula", "food"],
    "car": ["ride", "gari", "whip"],
    "house": ["keja", "hao", "place"],
    "phone": ["simu", "phone", "line"],
    "work": ["kazi", "hustle", "grind"],
    "school": ["shule", "sch", "chuo"],
    "problem": ["shida", "issue", "stress"],
    "thing": ["kitu", "thing", "vitu"],

    # Verbs (English base, used with Swahili prefixes)
    "go": ["enda", "go"],
    "come": ["kam", "come"],
    "eat": ["dishi", "eat"],
    "drink": ["drink", "nywa"],
    "sleep": ["lala", "sleep"],
    "talk": ["ongea", "talk", "chat"],
    "think": ["think", "fikiri"],
    "know": ["jua", "know"],
    "want": ["want", "taka"],
    "like": ["like", "penda"],
    "love": ["love", "penda"],

    # Adjectives
    "big": ["bigi", "kubwa"],
    "small": ["small", "ndogo"],
    "many": ["mob", "mengi"],
    "very": ["sana", "very", "mad"],
    "bad": ["mbaya", "bad", "mbovu"],
    "nice": ["fiti", "safi", "nice", "poa"],
    "fast": ["haraka", "fast", "speed"],
    "slow": ["pole pole", "slow"],
    "new": ["mpya", "new", "brand new"],
    "old": ["old", "ya zamani"],

    # Connectors & fillers
    "but": ["lakini", "but", "basi"],
    "so": ["so", "kwa hivyo", "basi"],
    "because": ["juu", "because", "coz", "maana"],
    "yes": ["eeh", "yes", "ndio", "aii"],
    "no": ["hapana", "no", "nah", "zi"],
    "okay": ["sawa", "okay", "iko sawa"],
    "right": ["right", "sawa", "ama?"],
    "really": ["kweli", "for real", "ati"],
    "now": ["saa hii", "now", "sahi"],
    "here": ["hapa", "here", "huku"],
    "there": ["pale", "there", "huko"],
    "today": ["leo", "today", "tuda"],
    "tomorrow": ["kesho", "tomoroh"],
    "please": ["tafadhali", "please", "basi"],
    "thanks": ["asante", "thanks", "sante"],
}

# Swahili verb prefixes for English verb stems
SW_VERB_PREFIXES = {
    "1sg_present": "na-",     # I am ___-ing: "na-drive"
    "1sg_past": "nili-",      # I ___-ed: "nili-enjoy"
    "2sg_present": "una-",    # You are ___-ing: "una-stress"
    "3sg_present": "ana-",    # He/she is ___-ing: "ana-text"
    "1pl_present": "tuna-",   # We are ___-ing: "tuna-plan"
    "3pl_present": "wana-",   # They are ___-ing: "wana-hustle"
    "3pl_past": "wali-",      # They ___-ed: "wali-cancel"
    "infinitive": "ku-",      # To ___: "ku-drive", "ku-enjoy"
    "3pl_perf": "wame-",      # They have ___-ed: "wame-cancel"
}

# English verbs commonly used with Swahili prefixes in Sheng
SHENG_VERB_STEMS = [
    "drive", "text", "call", "cancel", "enjoy", "plan", "stress",
    "hustle", "connect", "check", "push", "deal", "handle", "sort",
    "charge", "fix", "try", "help", "change", "move", "chill",
    "vibe", "link", "flex", "trip", "relate", "upgrade", "manage",
]

# Tag switches (appended or prepended to Swahili sentences)
TAG_SWITCHES = [
    ", right?", ", you know?", ", sawa?", ", ama?", ", si ndio?",
    ", for real", ", I swear", ", bro", ", manze", ", msee",
    "honestly, ", "like, ", "basically, ", "so basically, ",
]


def rule_based_codeswtich(
    text: str,
    lang: str = "sw",
    switch_prob: float = 0.3,
    tag_prob: float = 0.15,
    verb_prefix_prob: float = 0.1,
) -> str:
    """Apply rule-based code-switching to a sentence.

    Args:
        text: Input sentence (Swahili or English)
        lang: Source language ("sw" or "en")
        switch_prob: Probability of switching each eligible word
        tag_prob: Probability of adding a tag switch
        verb_prefix_prob: Probability of creating a Swahili-prefixed English verb
    """
    words = text.split()
    result = []

    for word in words:
        word_lower = word.lower().strip(".,!?;:")
        punctuation = ""
        if word and word[-1] in ".,!?;:":
            punctuation = word[-1]
            word_clean = word[:-1]
        else:
            word_clean = word

        switched = False

        # Try lexical substitution
        if random.random() < switch_prob:
            if word_lower in SHENG_LEXICON:
                replacement = random.choice(SHENG_LEXICON[word_lower])
                # Preserve capitalization of first letter
                if word_clean[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement + punctuation)
                switched = True

        # Try Swahili-prefixed English verb
        if not switched and random.random() < verb_prefix_prob:
            if word_lower in SHENG_VERB_STEMS:
                prefix_type = random.choice(list(SW_VERB_PREFIXES.keys()))
                prefix = SW_VERB_PREFIXES[prefix_type]
                result.append(f"{prefix}{word_lower}{punctuation}")
                switched = True

        if not switched:
            result.append(word)

    output = " ".join(result)

    # Maybe add a tag switch
    if random.random() < tag_prob:
        tag = random.choice(TAG_SWITCHES)
        if tag.endswith(" "):
            output = tag + output[0].lower() + output[1:]
        else:
            # Remove trailing period if adding a tag
            if output.endswith("."):
                output = output[:-1]
            output = output + tag

    return output


def generate_sheng_variants(
    text: str,
    n_variants: int = 3,
    lang: str = "sw",
) -> List[str]:
    """Generate multiple code-switched variants of a sentence using rules."""
    variants = set()
    attempts = 0
    max_attempts = n_variants * 5

    while len(variants) < n_variants and attempts < max_attempts:
        # Vary the intensity of code-switching
        switch_prob = random.uniform(0.15, 0.5)
        tag_prob = random.uniform(0.05, 0.25)
        verb_prob = random.uniform(0.05, 0.2)

        variant = rule_based_codeswtich(
            text, lang, switch_prob, tag_prob, verb_prob
        )
        if variant != text:
            variants.add(variant)
        attempts += 1

    return list(variants)


# ---------------------------------------------------------------------------
# LLM-based code-switching (Grok / Claude / OpenAI)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in Kenyan Sheng and East African code-switching patterns.
Your task is to create realistic code-switched variants of sentences, mixing
Swahili and English naturally the way young Kenyans speak in Nairobi.

Sheng characteristics:
- Mix English and Swahili words freely within sentences
- Use Swahili verb prefixes with English verb stems: "ana-drive", "ku-enjoy", "wame-cancel"
- Use Sheng slang: "sasa" (hello), "poa" (good), "msee" (man), "doh" (money), "keja" (house)
- Tag switches: ending sentences with "right?", "sawa?", "ama?"
- Informal tone, like texting or casual conversation
- NOT pidgin or broken English — Sheng is a legitimate dialect with consistent patterns

Examples:
- Clean Swahili: "Tutaenda sokoni kesho asubuhi" → Sheng: "Tuta-go soko kesho morning"
- Clean English: "I'll drive to work tomorrow" → Sheng: "Nita-drive to kazi tomorrow"
- Clean Swahili: "Habari za asubuhi" → Sheng: "Niaje, morning yako iko aje?"
"""


def build_augment_prompt(sentences: List[str], source_lang: str = "sw") -> str:
    """Build a prompt for LLM-based code-switch augmentation."""
    lang_name = "Swahili" if source_lang == "sw" else "English"

    prompt = f"""Generate 2 realistic Sheng/code-switched variants for each of these {lang_name} sentences.
Make each variant different in how much switching occurs (light vs heavy).
Return ONLY a JSON array of objects with "original", "light" (subtle switching), and "heavy" (more switching) keys.

Sentences:
"""
    for i, s in enumerate(sentences):
        prompt += f"{i+1}. {s}\n"

    prompt += "\nReturn valid JSON only, no explanation."
    return prompt


class LLMBackend:
    """Abstract LLM API interface."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GrokBackend(LLMBackend):
    """xAI Grok API backend."""

    def __init__(self, api_key: str, model: str = "grok-3-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.x.ai/v1"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.9,
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.9,
            },
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


class OpenAIBackend(LLMBackend):
    """OpenAI GPT API backend."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.9,
                "max_tokens": 4096,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def get_llm_backend(backend: str, api_key: str, model: str = None) -> LLMBackend:
    """Factory for LLM backends."""
    if backend == "grok":
        return GrokBackend(api_key, model=model or "grok-3-mini")
    elif backend == "claude":
        return ClaudeBackend(api_key, model=model or "claude-sonnet-4-20250514")
    elif backend == "openai":
        return OpenAIBackend(api_key, model=model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def llm_codeswtich_batch(
    sentences: List[str],
    llm: LLMBackend,
    source_lang: str = "sw",
    max_retries: int = 3,
) -> List[Dict]:
    """Generate code-switched variants for a batch of sentences via LLM.

    Returns list of dicts: {"original": str, "light": str, "heavy": str}
    """
    prompt = build_augment_prompt(sentences, source_lang)

    for attempt in range(max_retries):
        try:
            response = llm.generate(SYSTEM_PROMPT, prompt)

            # Extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = re.sub(r"```\w*\n?", "", json_str).strip()

            results = json.loads(json_str)

            if isinstance(results, list) and len(results) > 0:
                return results

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"  Failed to parse LLM response after {max_retries} attempts: {e}")

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                continue
            print(f"  LLM API error: {e}")

    # Return empty results on failure
    return [{"original": s, "light": s, "heavy": s} for s in sentences]


# ---------------------------------------------------------------------------
# Corpus generation (standalone Sheng text for tokenizer training)
# ---------------------------------------------------------------------------

CORPUS_PROMPT = """Generate {n} unique, realistic Sheng/code-switched sentences.
Each sentence should be something a young Kenyan might say in daily conversation.
Mix topics: greetings, directions, shopping, tech, school, work, relationships, news, sports.

Vary the intensity:
- Some sentences mostly Swahili with a few English words
- Some mostly English with Swahili slang
- Some heavily mixed (true Sheng)
- Include some with Swahili verb prefixes on English stems

Return ONLY a JSON array of strings. No numbering, no explanation."""


def generate_sheng_corpus(
    llm: LLMBackend,
    num_samples: int,
    output_file: str,
    batch_size: int = 50,
):
    """Generate a standalone Sheng text corpus for tokenizer training."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_sentences = []
    batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(batches):
        n = min(batch_size, num_samples - len(all_sentences))
        prompt = CORPUS_PROMPT.format(n=n)

        try:
            response = llm.generate(SYSTEM_PROMPT, prompt)
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = re.sub(r"```\w*\n?", "", json_str).strip()

            sentences = json.loads(json_str)
            if isinstance(sentences, list):
                all_sentences.extend(sentences)
                print(f"  Batch {batch_idx+1}/{batches}: got {len(sentences)} sentences "
                      f"(total: {len(all_sentences)})")
        except Exception as e:
            print(f"  Batch {batch_idx+1} failed: {e}")

        # Rate limiting
        time.sleep(0.5)

    # Write corpus
    with open(output_file, "w", encoding="utf-8") as f:
        for s in all_sentences:
            if isinstance(s, str) and s.strip():
                f.write(s.strip() + "\n")

    print(f"\nGenerated {len(all_sentences)} Sheng sentences -> {output_file}")
    return len(all_sentences)


# ---------------------------------------------------------------------------
# Main augmentation pipeline
# ---------------------------------------------------------------------------

def augment_translations(
    input_dir: str,
    output_dir: str,
    backend: str = "rules",
    llm: Optional[LLMBackend] = None,
    source_lang: str = "sw",
    max_samples: Optional[int] = None,
    batch_size: int = 20,
    n_variants_rules: int = 2,
    resume_from: int = 0,
) -> int:
    """Augment translation JSONs with code-switched variants.

    For each translation file, generates Sheng variants of the translated text
    and saves them as new translation files with "_sheng" suffix.
    """
    in_dir = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(
        f for f in in_dir.glob("*.json")
        if f.name != "index.jsonl"
    )

    if max_samples:
        json_files = json_files[:max_samples]

    print(f"Augmenting {len(json_files)} translations with {backend} backend")

    count = 0

    if backend == "rules":
        # Rule-based: process one at a time (fast, no API)
        for idx, jf in enumerate(json_files):
            if idx < resume_from:
                continue

            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            translated = data.get("translated_text", "").strip()
            if not translated:
                continue

            variants = generate_sheng_variants(
                translated, n_variants=n_variants_rules, lang=source_lang
            )

            for vi, variant in enumerate(variants):
                aug_data = {
                    **data,
                    "translated_text": variant,
                    "original_translation": translated,
                    "augmentation": "sheng_rules",
                    "variant_idx": vi,
                }
                out_name = jf.stem + f"_sheng{vi}" + jf.suffix
                out_path = os.path.join(output_dir, out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(aug_data, f, ensure_ascii=False, indent=2)

            count += len(variants)
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx+1}/{len(json_files)} ({count} variants)")

    else:
        # LLM-based: process in batches
        if llm is None:
            raise ValueError("LLM backend required for non-rules augmentation")

        batch_sentences = []
        batch_files = []

        for idx, jf in enumerate(json_files):
            if idx < resume_from:
                continue

            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            translated = data.get("translated_text", "").strip()
            if not translated:
                continue

            batch_sentences.append(translated)
            batch_files.append((jf, data))

            if len(batch_sentences) >= batch_size or idx == len(json_files) - 1:
                # Process batch
                results = llm_codeswtich_batch(
                    batch_sentences, llm, source_lang
                )

                for (jf_orig, orig_data), result in zip(batch_files, results):
                    for variant_key in ["light", "heavy"]:
                        variant_text = result.get(variant_key, "")
                        if variant_text and variant_text != orig_data.get("translated_text"):
                            aug_data = {
                                **orig_data,
                                "translated_text": variant_text,
                                "original_translation": orig_data["translated_text"],
                                "augmentation": f"sheng_{backend}_{variant_key}",
                                "variant_key": variant_key,
                            }
                            out_name = jf_orig.stem + f"_sheng_{variant_key}" + jf_orig.suffix
                            out_path = os.path.join(output_dir, out_name)
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(aug_data, f, ensure_ascii=False, indent=2)
                            count += 1

                batch_sentences = []
                batch_files = []

                if count % 500 == 0 and count > 0:
                    print(f"  Generated {count} variants so far...")

                # Rate limiting for API calls
                time.sleep(0.5)

    print(f"\nDone! Generated {count} code-switched variants -> {output_dir}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Sheng/code-switched text augmentations"
    )

    # Mode
    parser.add_argument("--mode", type=str, default="augment",
                        choices=["augment", "generate_corpus"],
                        help="'augment': transform existing translations. "
                             "'generate_corpus': create standalone Sheng text")

    # I/O
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Translation JSON directory (for augment mode)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")

    # Backend
    parser.add_argument("--backend", type=str, default="rules",
                        choices=["rules", "grok", "claude", "openai"],
                        help="Augmentation backend")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set GROK_API_KEY / ANTHROPIC_API_KEY / "
                             "OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model name")

    # Processing
    parser.add_argument("--source_lang", type=str, default="sw",
                        choices=["sw", "en"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for LLM API calls")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of sentences for corpus generation mode")
    parser.add_argument("--resume_from", type=int, default=0)

    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key
    if api_key is None and args.backend != "rules":
        env_map = {
            "grok": "GROK_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = env_map.get(args.backend)
        api_key = os.environ.get(env_var, "")
        if not api_key:
            parser.error(f"--api_key required (or set {env_var} env variable)")

    # Initialize LLM backend if needed
    llm = None
    if args.backend != "rules":
        llm = get_llm_backend(args.backend, api_key, args.model)

    if args.mode == "augment":
        if not args.input_dir:
            parser.error("--input_dir required for augment mode")
        augment_translations(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            backend=args.backend,
            llm=llm,
            source_lang=args.source_lang,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            resume_from=args.resume_from,
        )
    elif args.mode == "generate_corpus":
        if llm is None:
            parser.error("generate_corpus mode requires an LLM backend (not rules)")
        output_file = os.path.join(args.output_dir, "sheng_corpus.txt")
        generate_sheng_corpus(
            llm=llm,
            num_samples=args.num_samples,
            output_file=output_file,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
