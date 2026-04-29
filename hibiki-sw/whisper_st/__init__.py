"""Transcript-prompted speech translation experiment for Sw<->En.

Architecture inspired by Wakanda AI (Gichamba et al., Afretec 2025):
- Whisper encoder + CTC head produces source-language transcript hypothesis
- Decoder conditions on (audio features + transcript + lexicon hints) to translate

Our delta: lexicon augmentation. Named entities and Kenyan-specific terms are a
known weakness of NLLB on Kenyan Swahili; injecting a small Sw<->En lexicon
into the decoder prompt closes that gap.
"""
