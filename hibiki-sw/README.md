# Training a 100M Parameter Hibiki-Style Bidirectional En↔Sw Speech Translation Model on Kaggle

## Context

The goal is to train a ~100M parameter end-to-end speech-to-speech translation model supporting **both English → Swahili and Swahili → English**, inspired by the Hibiki architecture (Kyutai, 2025). Hibiki uses a decoder-only multistream transformer that synchronously processes source and target speech via a Temporal Transformer + Depth Transformer, leveraging the Mimi neural audio codec for tokenization. The original Hibiki is 2.7B params (French→English); we must scale down to ~100M and train on Kaggle's 2× T4 GPUs (16GB VRAM each) with a **30hr/week GPU quota** (60hrs over 2 weeks) and 9hr max session length.

**Key challenge:** English-Swahili is a low-resource pair — no large-scale parallel speech translation dataset exists. We must generate synthetic parallel data for both directions.

**Bidirectional strategy:** Stages 0-2 (VITS training, text adaptation, audio pretraining) are **shared** — they are language-agnostic. Stages 3-4 (S2ST training + fine-tuning) are run **twice** — once per direction — branching from the same Stage 2 checkpoint.

---

## 1. Architecture Design (~100M Parameters)

### 1.1 Neural Audio Codec: Mimi (Frozen, pretrained)
- Use the pretrained **`kyutai/mimi`** codec from HuggingFace (CC-BY license)
- 12.5 Hz frame rate, **Q=8 codebooks** (reduced from 16 to cut depth transformer cost)
- Codec is **frozen** — not counted in the 100M param budget
- Encodes 24kHz mono audio → discrete tokens; codebook 1 = semantic, codebooks 2-8 = acoustic

### 1.2 Temporal Transformer (Main backbone, ~65M params)
| Hyperparameter | Value |
|---|---|
| Latent dimension (d_model) | 512 |
| FFN inner dim (gated SiLU) | 1408 |
| Layers | 12 |
| Attention heads | 8 |
| Head dim | 64 |
| Context window | 250 tokens (~20s at 12.5Hz) |
| Local attention window | 250 tokens |

- Per-layer params: self-attn (4 × 512²) + gated-FFN (3 × 512 × 1408) ≈ 3.2M
- 12 layers ≈ 38.4M + embeddings + output heads ≈ **~65M total**

### 1.3 Depth Transformer (~30M params)
| Hyperparameter | Value |
|---|---|
| Latent dimension | 384 |
| FFN inner dim (gated SiLU) | 1024 |
| Layers per codebook step | 4 |
| Codebook levels (output stream) | 8 |
| Codebook levels (input stream, train only) | 8 |
| Weight sharing | Codebooks 5-8 share weights |

- Per-layer: ~1.7M params × 4 layers = 6.8M base
- With separate weights for codebooks 1-4, shared for 5-8 → ~5 weight sets × 6.8M ≈ **~30M** (with embeddings)
- Low-rank embedding tables (dim 64) for acoustic tokens to save parameters

### 1.4 Token Streams (following Hibiki)
- **Source stream (listens):** Mimi tokens from English input audio (ground truth at train, live at inference)
- **Hibiki stream (speaks):** Mimi tokens for Swahili output audio (predicted)
- **Inner Monologue text stream:** Swahili text tokens aligned with output audio (predicted)
- Acoustic delay of 2 timesteps between semantic and acoustic tokens (per Hibiki eq. 3)

### 1.5 Total Parameter Count
| Component | Params |
|---|---|
| Temporal Transformer | ~65M |
| Depth Transformer | ~30M |
| Embeddings + heads | ~5M |
| **Total** | **~100M** |

---

## 2. Data Pipeline

### 2.1 Available Datasets

| Dataset | Type | Size | Use |
|---|---|---|---|
| **Common Voice (Swahili)** | Swahili read speech + transcripts | ~300hrs validated | Audio pretraining, TTS source |
| **Common Voice (English)** | English read speech + transcripts | 2000+ hrs | Audio pretraining, source audio |
| **FLEURS (en/sw)** | Parallel read speech, 102 langs | ~10hrs per lang | **Evaluation benchmark** |
| **WAXAL (Swahili)** | Swahili speech (Google, 2026) | ~1250hrs transcribed | Audio pretraining |
| **OPUS/CCAligned en-sw** | Parallel text | ~230K sentence pairs | Text pretraining, translation data |
| **MADLAD-400 3B** | Translation model | - | Generating synthetic translations |

### 2.2 Synthetic Parallel Speech Data Generation (offline, CPU/Colab)

Since no large En↔Sw parallel speech corpus exists, we follow Hibiki's approach **for both directions:**

#### Direction 1: En→Sw
1. **Source audio (English):** Use Common Voice English utterances (~500hrs, clean single-speaker, 10-30s)
2. **Transcribe:** Whisper `small`/`medium` → English transcripts with word timestamps
3. **Translate:** MADLAD-400-3B (`google/madlad400-3b-mt`) En→Sw sentence by sentence
4. **Synthesize Swahili speech:** Fine-tuned VITS on CV-Swahili, conditioned on reference speaker
5. **Contextual alignment:** Per-word alignment via MADLAD perplexity (eq. 6)
6. **Silence insertion:** Insert silences into Swahili audio for causal alignment

#### Direction 2: Sw→En
1. **Source audio (Swahili):** Use Common Voice Swahili (~300hrs) + WAXAL Swahili if available
2. **Transcribe:** Whisper `small`/`medium` → Swahili transcripts with word timestamps
3. **Translate:** MADLAD-400-3B Sw→En sentence by sentence
4. **Synthesize English speech:** Use a pretrained English TTS (e.g., Coqui VITS-LJSpeech, or any high-quality English TTS — no training needed)
5. **Contextual alignment:** Same method, reversed direction
6. **Silence insertion:** Insert silences into English audio for causal alignment

**TTS: Fine-tune VITS on Common Voice Swahili**
- Train a VITS model on Common Voice Swahili (~300hrs validated) — **Stage 0** of training, ~4 GPU-hrs on Kaggle
- VITS gives natural prosody and can be conditioned on speaker embeddings for voice cloning
- Use Coqui TTS framework (or the `vits` package) for training
- After training, use VITS to synthesize Swahili audio for all translated transcripts
- Synthesis itself can run on CPU (slow but free) or batch on GPU

**Target:** Generate ~200-500hrs of aligned En→Sw parallel speech pairs.

Contextual alignment computation and silence insertion are done **offline** (not on Kaggle GPU quota) using CPU.

### 2.3 Data Preprocessing
- Encode all audio through Mimi codec → store discrete tokens (much smaller than raw audio)
- Pre-tokenize all text with a SentencePiece/BPE tokenizer (vocab ~32K, trained on en+sw text)
- Store as memory-mapped numpy arrays or HuggingFace datasets for fast loading
- Upload preprocessed token datasets to Kaggle as a dataset (~few GB)

---

## 3. Training Protocol

### 3.1 Overview of Training Stages

| Stage | Data | Steps | Batch | LR | Est. GPU Hours |
|---|---|---|---|---|---|
| 0. VITS TTS Training | Common Voice Swahili | ~50K | 16 | 2e-4 | ~4hrs |
| 1. Text Adaptation | en+sw text (OPUS, CC) | 15K | 256 seq × 1024 tok | 1e-4 | ~2hrs |
| 2. Audio Pretraining | en+sw monolingual audio | 80K | 64 | 1.5e-4 | ~12hrs |
| **--- Shared stages above, direction-specific below ---** | | | | | |
| 3a. S2ST Training (En→Sw) | Synthetic En→Sw pairs | 40K | 32 | 2e-5 | ~11hrs |
| 4a. Fine-tuning (En→Sw) | Curated En→Sw subset | 4K | 8 | 1e-6 | ~2hrs |
| 3b. S2ST Training (Sw→En) | Synthetic Sw→En pairs | 40K | 32 | 2e-5 | ~11hrs |
| 4b. Fine-tuning (Sw→En) | Curated Sw→En subset | 4K | 8 | 1e-6 | ~2hrs |
| **Total** | | | | | **~44hrs** |

This fits within the 2-week / 60hr GPU budget, with ~16hrs buffer for debugging, ablations, and evaluation.

**Note:** Stages 3a/4a and 3b/4b branch from the **same Stage 2 checkpoint** and produce two separate model files. They can be trained sequentially across Kaggle sessions.

### 3.2 Stage 1: Text Adaptation (~2 GPU-hrs)

- **Initialize Temporal Transformer from a pretrained small LM** (e.g., `sberbank-ai/mGPT` 100M-scale, or a distilled multilingual causal LM). This skips cold-start text pretraining and saves ~2-3hrs.
- Adapt on **English + Swahili text** from OPUS parallel corpus, CCAligned, and Wikipedia
- Brief continued pretraining: 15K steps (vs. Hibiki's 600K from scratch)
- Sequence length: 1024 tokens
- Optimizer: AdamW (β1=0.9, β2=0.95, weight decay=0.1)
- Cosine LR schedule with 500 warmup steps, peak 1e-4 (lower than from-scratch since pretrained)
- **Mixed precision (FP16)** via PyTorch AMP
- **2-GPU DDP** via `torch.nn.parallel.DistributedDataParallel`
- Effective batch size 256 via gradient accumulation (local batch 16 × 8 accum steps × 2 GPUs)

### 3.3 Stage 2: Audio Pretraining (~12 GPU-hrs)

- Continue from text-pretrained Temporal Transformer
- Add Depth Transformer (randomly initialized)
- Train on **non-parallel** English and Swahili audio (single-stream, as in Moshi/Hibiki)
- Each sample: one language's audio encoded via Mimi, predict next tokens (semantic + acoustic)
- After this stage, **duplicate Depth Transformer weights** for multistream modeling
- Cosine LR 1.5e-4, batch size 64 (16 × 2 accum × 2 GPUs)
- Loss on all Q=8 codebook levels

### 3.4 Stage 3: Speech Translation Training (~11 GPU-hrs per direction)

Run **twice** from the same Stage 2 checkpoint — once for En→Sw, once for Sw→En.

- Switch to **multistream** mode: source + target audio concatenated along codebook axis
- Use synthetic parallel data with silence-inserted contextual alignment
- **En→Sw direction:** source=English audio, target=Swahili audio (synthesized by VITS)
- **Sw→En direction:** source=Swahili audio, target=English audio (synthesized by any English TTS, e.g., pretrained VITS-LJSpeech or Bark)
- Predict:
  - Target audio tokens (semantic + 7 acoustic)
  - Source audio tokens (input reconstruction, training only)
  - Target Inner Monologue text stream
- Conditional training on speaker similarity label (5 categories) — add learnable embedding per category
- Source audio noise augmentation (Gaussian noise, speed perturbation)
- LR: 2e-5, batch size 32 (8 × 2 accum × 2 GPUs)
- Loss on both source and target streams
- EOS tokens on source (end of source speech) and target text stream

### 3.5 Stage 4: Fine-tuning (~2 GPU-hrs per direction)

Run **twice** — once per direction, continuing from the respective Stage 3 checkpoint.

- Use a curated high-quality subset: best aligned generations with high speaker similarity
- ~50hrs of data per direction
- LR: 1e-6, batch 8
- 4K steps
- Classifier-free guidance training: randomly drop speaker similarity label with 20% probability

---

## 4. Kaggle-Specific Optimizations

### 4.1 Memory Management (2× T4 = 2× 16GB VRAM)
- **FP16 mixed precision** throughout (T4 has good FP16 throughput)
- **Gradient checkpointing** on the Temporal Transformer (saves ~40% memory, ~20% slower)
- **Gradient accumulation** to achieve target effective batch sizes
- Pre-tokenized data (Mimi tokens stored as uint16) to minimize CPU→GPU transfer
- Pin memory, num_workers=2 per GPU for data loading

### 4.2 Multi-GPU Strategy
- **DDP (DistributedDataParallel)** — simplest, works well for 100M model that fits on one GPU
- No model parallelism needed (100M params ≈ 200MB in FP16, fits easily)
- Each GPU holds full model + optimizer states (~800MB in FP16 with AdamW)

### 4.3 Checkpointing & Session Management
- **Save checkpoint every 2K steps** to `/kaggle/working/` (persists across sessions)
- Upload Stage 2 checkpoint to **HuggingFace Hub** (private repo) for cross-account transfer
- Each Kaggle session: 9hr max → plan training stages to fit within sessions

#### 2-Account Parallel Strategy

**Account A** runs shared pretraining + En→Sw direction. **Account B** runs Sw→En direction (starts after Stage 2 checkpoint is available from A).

| Session | Account A | Account B |
|---|---|---|
| A-1 (~6hrs) | Stage 0 (VITS) + Stage 1 (text adapt) | *idle — waiting for checkpoint* |
| A-2 (~9hrs) | Stage 2 (audio pretraining) | *idle* |
| A-3 (~6hrs) | Finish Stage 2 + Mimi encode + **upload ckpt to HF Hub** | Download Stage 2 ckpt from HF Hub |
| A-4 (~9hrs) | Stage 3a — En→Sw S2ST training | Stage 3b — Sw→En S2ST training |
| A-5 (~4hrs) | Finish 3a + Stage 4a fine-tune + eval | Finish 3b + Stage 4b fine-tune + eval |

- **Account A total:** ~34hrs (within 30hr/week × 1 week, may spill 4hrs into week 2)
- **Account B total:** ~13hrs
- **Wall-clock time: ~5-6 days** (vs. 2 weeks with 1 account)
- **Buffer: ~23hrs combined** for debugging, ablations, re-runs

### 4.4 Efficient Data Loading
- Upload pre-tokenized dataset as a Kaggle Dataset (avoids re-processing)
- Use memory-mapped files (np.memmap) for token sequences
- Lazy loading with prefetch

---

## 5. Repository Structure

```
hibiki-sw/
├── configs/
│   └── model_100m.yaml              # All hyperparameters
├── data/
│   ├── dataset.py                    # PyTorch Dataset classes (Text, Audio, S2ST)
│   ├── contextual_align.py           # Contextual alignment with MADLAD perplexity
│   ├── silence_insertion.py          # Insert silences for causal alignment
│   └── prepare/
│       ├── train_tokenizer.py        # Train SentencePiece BPE tokenizer (en+sw)
│       ├── tokenize_text.py          # Tokenize text data for Stage 1
│       ├── encode_audio.py           # Encode audio → Mimi codec tokens (.npy)
│       └── create_s2st_manifest.py   # Create TSV manifests for Stages 3-4
├── model/
│   ├── temporal_transformer.py       # Temporal (global) transformer with RoPE
│   ├── depth_transformer.py          # Depth (local) transformer
│   ├── hibiki_model.py               # Full Hibiki model combining both
│   └── codec.py                      # Mimi codec wrapper (frozen)
├── training/
│   ├── train_text.py                 # Stage 1: text adaptation
│   ├── train_audio.py                # Stage 2: audio pretraining
│   ├── train_s2st.py                 # Stage 3: speech translation
│   ├── train_finetune.py             # Stage 4: fine-tuning with CFG
│   └── utils.py                      # DDP, checkpointing, logging, schedulers
├── inference/
│   └── translate.py                  # End-to-end translation pipeline
├── evaluation/
│   └── evaluate.py                   # ASR-BLEU, speaker similarity, latency
├── notebooks/
│   ├── 00_data_preparation.ipynb     # Data prep (~2hr GPU)
│   ├── 01_stage1_text.ipynb          # Text adaptation (~4hr GPU)
│   ├── 02_stage2_audio.ipynb         # Audio pretraining (~12hr, multi-session)
│   ├── 03_stage3_s2st.ipynb          # S2ST training (~8hr GPU)
│   └── 04_stage4_finetune.ipynb      # Fine-tune + evaluation (~3hr GPU)
└── requirements.txt
```

---

## 6. Evaluation Plan

### 6.1 Metrics
- **ASR-BLEU:** Transcribe Swahili output with Whisper → compute BLEU against reference translation
- **Text BLEU:** From Inner Monologue text stream directly
- **Speaker similarity:** Cosine similarity of WavLM speaker embeddings (source vs. output)
- **LAAL (latency):** Length-Adaptive Average Lagging
- **End Offset:** Time between last source word and last output word

### 6.2 Evaluation Data
- **FLEURS en↔sw test set** (~10hrs per language, parallel read speech with reference translations)
- Evaluate **both directions** on the same benchmark
- **Audio-NTREX** style: if time permits, record/collect a small long-form test set

### 6.3 Baselines to Compare Against
- **Seamless (Meta):** Supports en↔sw, available on HuggingFace
- **Cascaded pipeline En→Sw:** Whisper ASR → MADLAD-400 MT → VITS-Swahili
- **Cascaded pipeline Sw→En:** Whisper ASR → MADLAD-400 MT → English TTS

---

## 7. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Swahili TTS quality is poor | Use MMS-TTS-swh (Meta) which covers Swahili; alternatively fine-tune VITS on Common Voice Swahili |
| 100M model too small for quality | Focus on shorter utterances (10-20s), reduce context window, accept lower BLEU than Hibiki |
| Synthetic data quality issues | Filter by ASR round-trip consistency (translate back and check similarity) |
| Mimi codec not great for Swahili | Mimi was trained on speech generally; evaluate reconstruction quality first. Fall back to EnCodec if needed |
| 60hr budget still tight | Pre-tokenize everything offline; use pretrained LM init (saves ~2hrs); gradient accumulation; 25hr buffer for issues |
| Pretrained LM dimension mismatch | If no exact 512-dim pretrained LM exists, use weight projection or train a thin adapter layer on top |
| Contextual alignment errors | Use conservative 2s minimum lag buffer (as in Hibiki); smooth out spike anomalies |

---

## 8. Dependencies

```
torch>=2.0
transformers>=4.36  # For Mimi codec, MADLAD-400
accelerate
sentencepiece
datasets
whisper / faster-whisper
scipy
numpy
soundfile
```

---

## 9. Week-by-Week Timeline

### Pre-work (before GPU sessions, ~1-2 weeks, CPU only)
- [ ] Build and unit-test model architecture on CPU
- [ ] **En→Sw data:** Select ~500hrs Common Voice English, transcribe with Whisper, translate En→Sw with MADLAD-400-3B
- [ ] **Sw→En data:** Select ~300hrs Common Voice Swahili, transcribe with Whisper, translate Sw→En with MADLAD-400-3B
- [ ] Compute contextual alignments for both directions (MADLAD perplexity, eq. 6)
- [ ] Prepare Common Voice Swahili data for VITS training
- [ ] Prepare en+sw text corpus for Stage 1
- [ ] Upload all preprocessed data to **both Kaggle accounts** as datasets

### Week 1 — Account A (shared stages + En→Sw)
- [ ] A-1 (~6hrs): Train VITS TTS on CV-Swahili + Text adaptation
- [ ] A-2 (~9hrs): Audio pretraining (single-stream, en+sw monolingual)
- [ ] A-3 (~6hrs): Finish audio pretraining + Mimi encoding + **upload Stage 2 ckpt to HF Hub**
- [ ] A-4 (~9hrs): En→Sw S2ST training (Stage 3a)
- [ ] A-5 (~4hrs): En→Sw fine-tuning (Stage 4a) + evaluation

### Week 1 — Account B (Sw→En, starts after Stage 2 ckpt ready)
- [ ] B-1 (~9hrs): Download Stage 2 ckpt → Sw→En S2ST training (Stage 3b)
- [ ] B-2 (~4hrs): Finish Sw→En S2ST + fine-tuning (Stage 4b) + evaluation

### Post-training (either account)
- [ ] Evaluate both directions on FLEURS en↔sw (ASR-BLEU, speaker sim, LAAL)
- [ ] Compare against cascaded pipeline (Whisper → MADLAD → TTS) and Seamless
- [ ] Generate demo audio samples for project presentation
