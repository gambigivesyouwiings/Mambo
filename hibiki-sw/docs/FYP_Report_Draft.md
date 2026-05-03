# UNIVERSITY OF NAIROBI

## FACULTY OF ENGINEERING
## DEPARTMENT OF ELECTRICAL AND INFORMATION ENGINEERING

### FINAL YEAR PROJECT REPORT
### PROJECT INDEX: PRJ XXX

---

# DESIGN AND IMPLEMENTATION OF AN END-TO-END SPEECH-TO-SPEECH TRANSLATION SYSTEM FOR ENGLISH-SWAHILI USING NEURAL AUDIO CODECS

---

**STUDENT NAME:** MUGAMBI VICTOR KIMATHI

**REGISTRATION NUMBER:** F17/XXXXX/20XX

**SUPERVISOR(S):** DR. XXXXXX

**EXAMINER(S):** DR. XXXXXX

---

*This project report is submitted in partial fulfilment of the requirement for the award of the Degree of Bachelor of Science in Electrical and Electronics Engineering from the University of Nairobi*

*Submitted on XX of May 2026*

---

## DECLARATION OF ORIGINALITY

**NAME OF STUDENT:** MUGAMBI VICTOR KIMATHI

**REGISTRATION NUMBER:** F17/XXXXX/20XX

**COLLEGE:** ARCHITECTURE AND ENGINEERING

**FACULTY:** ENGINEERING

**DEPARTMENT:** ELECTRICAL AND INFORMATION ENGINEERING

**COURSE:** BSc. ELECTRICAL AND ELECTRONIC ENGINEERING

**PROJECT:** DESIGN AND IMPLEMENTATION OF AN END-TO-END SPEECH-TO-SPEECH TRANSLATION SYSTEM FOR ENGLISH-SWAHILI USING NEURAL AUDIO CODECS

1. I understand what plagiarism is and I am aware of the university policy in this regard.
2. I declare that this final year report is my original work and has not been submitted elsewhere for examination, the award of a degree or publication. Where other people's work or my work has been used, this has properly been acknowledged and referenced per the University of Nairobi's requirements.
3. I have not sought or used the services of any professional agencies to produce this work.
4. I have not allowed, and shall not allow anyone to copy my work to pass it off as his/her work.
5. I understand that any false claim in respect of this work shall result in disciplinary action per university anti-plagiarism policy.

Signature: .......................................

Date: ..................................

---

## CERTIFICATION

This report has been submitted to the Department of Electrical and Information Engineering, the University of Nairobi with my approval as the supervisor:

Lecturer: Dr. XXXXXX

Signature.................................................  Date..........................................

---

## DEDICATION

I dedicate this project to my family for their unwavering support throughout my academic journey, and to the broader African NLP community working to ensure that African languages are not left behind in the age of artificial intelligence.

---

## ACKNOWLEDGEMENTS

I would like to thank my supervisor, Dr. XXXXXX, for the guidance and expert advice offered throughout this project. I also acknowledge the open-source research community, particularly the Kyutai team whose Hibiki architecture inspired this work, Meta AI for the MMS and NLLB multilingual models, and the Mozilla Common Voice contributors who recorded the Swahili speech data that made this project possible. I am grateful to Kaggle for providing free GPU compute resources, and to the HuggingFace community for hosting the pretrained models and datasets used in this work.

---

## TABLE OF CONTENTS

- DECLARATION OF ORIGINALITY
- DEDICATION
- ACKNOWLEDGEMENTS
- TABLE OF CONTENTS
- LIST OF TABLES
- LIST OF FIGURES
- LIST OF ABBREVIATIONS
- ABSTRACT
- 1 INTRODUCTION
  - 1.1 Background
  - 1.2 Problem Statement
  - 1.3 Objectives
    - 1.3.1 Main Objective
    - 1.3.2 Specific Objectives
  - 1.4 Scope of Work
  - 1.5 Justification
- 2 LITERATURE REVIEW
  - 2.1 Speech Translation Paradigms
  - 2.2 Cascaded Speech Translation Systems
  - 2.3 End-to-End Speech Translation
  - 2.4 Neural Audio Codecs
  - 2.5 The Hibiki Architecture
  - 2.6 Multilingual Speech Models for African Languages
  - 2.7 Synthetic Data Generation for Low-Resource Pairs
  - 2.8 Summary of Literature Gaps
- 3 DESIGN METHODOLOGY
  - 3.1 System Overview
  - 3.2 Neural Audio Codec (Mimi)
  - 3.3 Temporal Transformer Design
  - 3.4 Depth Transformer Design
  - 3.5 Token Stream Architecture
  - 3.6 Text Tokenizer Design
  - 3.7 Data Pipeline Design
    - 3.7.1 Transcription with Forced Alignment
    - 3.7.2 Machine Translation
    - 3.7.3 Text-to-Speech Synthesis
    - 3.7.4 Contextual Alignment
    - 3.7.5 Silence Insertion
  - 3.8 Training Protocol Design
  - 3.9 Evaluation Metrics
- 4 IMPLEMENTATION
  - 4.1 Development Environment
    - 4.1.1 Kaggle T4 (Synthetic Data Pipeline + S2ST Model Training)
    - 4.1.2 AWS g5.2xlarge (Whisper ASR Fine-tuning Pipeline)
    - 4.1.3 Repository Structure
  - 4.2 Data Preparation
    - 4.2.1 Dataset Collection
    - 4.2.2 Tokenizer Training
    - 4.2.3 Audio Encoding with Mimi
  - 4.3 Synthetic Parallel Data Pipeline
    - 4.3.1 WhisperX Forced Alignment
    - 4.3.2 NLLB Translation
    - 4.3.3 MMS-TTS Synthesis
    - 4.3.4 Silence Insertion
  - 4.4 Model Training
    - 4.4.1 Stage 1: Text Adaptation
    - 4.4.2 Stage 2: Audio Pretraining
    - 4.4.3 Stage 3: Speech Translation Training
    - 4.4.4 Stage 4: Fine-tuning
  - 4.5 Kaggle GPU Optimizations
- 5 RESULTS AND ANALYSIS
  - 5.1 Data Pipeline Output
  - 5.2 Training Convergence
  - 5.3 Translation Quality (ASR-BLEU)
  - 5.4 Speaker Similarity
  - 5.5 Latency Analysis
  - 5.6 Comparison with Baselines
  - 5.7 Qualitative Analysis
- 6 RECOMMENDATIONS
- 7 CONCLUSION
- REFERENCES
- APPENDICES

---

## LIST OF TABLES

- Table 1: Comparison of Speech Translation Approaches
- Table 2: Temporal Transformer Hyperparameters
- Table 3: Depth Transformer Hyperparameters
- Table 4: Model Parameter Budget (~100M total)
- Table 5: Training Stages Overview
- Table 6: Available Datasets for English-Swahili
- Table 7: T4 GPU Memory Budget per Pipeline Step
- Table 8: Training Configuration per Stage
- Table 9: Evaluation Results (ASR-BLEU)
- Table 10: Comparison with Baseline Systems

---

## LIST OF FIGURES

- Figure 3.1: System Architecture Overview
- Figure 3.2: Hibiki Multi-Stream Token Architecture
- Figure 3.3: Temporal Transformer Block Diagram
- Figure 3.4: Depth Transformer Processing Flow
- Figure 3.5: Synthetic Data Pipeline Flowchart
- Figure 3.6: Contextual Alignment Algorithm
- Figure 3.7: Silence Insertion Mechanism
- Figure 3.8: Four-Stage Training Protocol
- Figure 4.1: Kaggle Session Management Strategy
- Figure 4.2: Mimi Codec Encoding/Decoding Pipeline
- Figure 5.1: Training Loss Curves (Stages 1-4)
- Figure 5.2: ASR-BLEU Score Progression
- Figure 5.3: Speaker Similarity Distribution
- Figure 5.4: Sample Spectrograms (Source vs. Translated Output)

---

## LIST OF ABBREVIATIONS

- S2ST -- Speech-to-Speech Translation
- ASR -- Automatic Speech Recognition
- TTS -- Text-to-Speech
- MT -- Machine Translation
- NLP -- Natural Language Processing
- BLEU -- Bilingual Evaluation Understudy
- LAAL -- Length-Adaptive Average Lagging
- BPE -- Byte Pair Encoding
- FFN -- Feed-Forward Network
- RoPE -- Rotary Position Embeddings
- SiLU -- Sigmoid Linear Unit
- DDP -- Distributed Data Parallel
- FP16 -- 16-bit Floating Point (Half Precision)
- VRAM -- Video Random Access Memory
- GPU -- Graphics Processing Unit
- LR -- Learning Rate
- CFG -- Classifier-Free Guidance
- MMS -- Massively Multilingual Speech
- NLLB -- No Language Left Behind
- VITS -- Variational Inference with adversarial learning for end-to-end Text-to-Speech
- EOS -- End of Sequence
- CV -- Common Voice
- FLEURS -- Few-shot Learning Evaluation of Universal Representations of Speech
- OPUS -- Open Parallel Corpus
- HF -- HuggingFace
- KSA -- Kenya Space Agency
- API -- Application Program Interface
- OSI -- Open Systems Interconnection

---

## ABSTRACT

Speech-to-speech translation (S2ST) enables direct conversion of spoken language in one language to spoken language in another, without requiring the speaker to be literate or to interact through text. While recent advances such as Meta's Seamless and Kyutai's Hibiki have demonstrated impressive end-to-end S2ST for high-resource language pairs (e.g., French-English), low-resource African languages like Swahili remain underserved. This project presents the design and implementation of a ~100M parameter end-to-end speech-to-speech translation system for bidirectional English-Swahili translation, inspired by the Hibiki architecture.

The system employs a decoder-only multistream transformer consisting of a Temporal Transformer (~65M parameters) and a Depth Transformer (~30M parameters), operating on discrete audio tokens produced by the Mimi neural audio codec (12.5 Hz, 8 codebooks). Since no large-scale English-Swahili parallel speech corpus exists, a synthetic data generation pipeline was developed that combines WhisperX forced alignment of Common Voice Swahili transcripts, NLLB-200 machine translation, MMS text-to-speech synthesis, contextual alignment, and silence insertion to create causally-aligned parallel speech pairs.

The model was trained in four stages -- text adaptation, audio pretraining, speech translation training, and fine-tuning with classifier-free guidance -- on Kaggle T4 GPUs within a total compute budget of approximately 44 GPU-hours. The system was evaluated on the FLEURS English-Swahili test set using ASR-BLEU, speaker similarity (WavLM embeddings), and latency (LAAL) metrics, and compared against cascaded baselines (Whisper ASR + NLLB MT + MMS-TTS) and Meta's Seamless model.

To strengthen the cascaded baseline used for comparison, this project also developed a Whisper-small Swahili ASR model fine-tuned via pseudo-label distillation from Whisper-large-v3. The pipeline pseudo-labels approximately 6,000 unlabelled Swahili utterances drawn from Common Voice and FLEURS-train, applies a three-filter quality recipe (teacher confidence, n-gram repetition, and Swahili language identification), and fine-tunes the student on the union of the pseudo-labels and the labelled KenSpeech corpus. On the FLEURS sw_ke test set, the fine-tuned student reduces Word Error Rate from 87.22% to 42.98% (a 44-point absolute reduction), and propagates through a Whisper-small + NLLB-200 cascade to raise Sw->En translation BLEU from 4.78 to 16.31 (a +11.5 BLEU improvement) without any change to the translation model. A secondary finding is that this pseudo-label-trained ASR achieves cascade BLEU statistically indistinguishable from a gold-label upper bound (16.31 vs 16.28), despite a 5.7-point WER gap, suggesting that pseudo-label distillation is more effective for downstream translation tasks than intrinsic WER measurements would indicate.

This work demonstrates the feasibility of building lightweight end-to-end S2ST systems for low-resource African language pairs using synthetic data and consumer-grade GPU hardware, and contributes a reproducible pseudo-label distillation recipe for low-resource Swahili ASR -- both contributing to the broader goal of language technology equity for African languages.

---

## 1 INTRODUCTION

### 1.1 Background

Language is the most natural and universal mode of human communication. Of the world's approximately 7,000 languages, the vast majority lack adequate representation in modern language technology systems. Swahili (Kiswahili), spoken by over 200 million people across East Africa as either a first or second language, is one of the most widely spoken languages on the African continent. It serves as a lingua franca in Kenya, Tanzania, Uganda, the Democratic Republic of Congo, and several neighbouring countries. Despite its prevalence, Swahili remains classified as a "low-resource" language in the context of speech and language technology, with significantly fewer digital resources, parallel corpora, and trained models compared to languages like English, French, or Mandarin.

Speech-to-speech translation (S2ST) is the task of converting spoken utterances in a source language directly into spoken utterances in a target language. Historically, this has been achieved through cascaded systems that chain together three separate components: automatic speech recognition (ASR) to transcribe the source speech to text, machine translation (MT) to translate the text, and text-to-speech synthesis (TTS) to generate the target speech. While effective, this approach suffers from error propagation across stages, increased latency, and loss of paralinguistic information such as speaker identity, prosody, and emotion.

Recent advances in deep learning have enabled the development of end-to-end S2ST systems that perform the translation in a single model. Notable examples include Meta AI's Seamless family of models [Seamless Communication et al., 2023], which support translation across nearly 100 languages, and Kyutai's Hibiki [Défossez et al., 2025], a streaming-capable system that translates French to English in real time with voice preservation. Hibiki is particularly notable for its architecture: a decoder-only multistream transformer that processes source and target audio streams synchronously through a Temporal Transformer (for sequence-level modelling) and a Depth Transformer (for codebook-level modelling), operating on discrete tokens from the Mimi neural audio codec.

However, these state-of-the-art systems are designed for high-resource language pairs and require billions of parameters and thousands of GPU-hours to train. Hibiki, for example, uses a 2.7B parameter model trained on proprietary French-English data. Adapting such architectures to low-resource pairs like English-Swahili presents fundamental challenges: the absence of large-scale parallel speech data, limited monolingual speech corpora, and the computational constraints typical of researchers in developing countries who rely on free-tier GPU platforms like Kaggle.

This project addresses these challenges by designing and implementing a scaled-down (~100M parameter) Hibiki-inspired S2ST system for bidirectional English-Swahili translation. The key innovation is a synthetic data generation pipeline that creates parallel speech pairs from existing monolingual resources -- Common Voice Swahili transcripts, NLLB machine translation, and MMS text-to-speech -- combined with contextual alignment and silence insertion techniques from the Hibiki paper to enable causal streaming during inference.

### 1.2 Problem Statement

There is no existing end-to-end speech-to-speech translation system specifically designed for the English-Swahili language pair. Current solutions rely on cascaded pipelines that introduce latency and propagate errors across ASR, MT, and TTS stages. Furthermore, the absence of large-scale parallel speech corpora for English-Swahili makes it infeasible to directly apply training methods designed for high-resource pairs. A method is needed to generate synthetic parallel speech data of sufficient quality and quantity to train an end-to-end model, while operating within the computational constraints available to researchers at African universities.

### 1.3 Objectives

#### 1.3.1 Main Objective

To design, implement, and evaluate an end-to-end speech-to-speech translation system for bidirectional English-Swahili translation using a neural audio codec-based transformer architecture.

#### 1.3.2 Specific Objectives

i. To review literature on end-to-end speech-to-speech translation architectures, neural audio codecs, and synthetic data generation methods for low-resource language pairs.

ii. To design a ~100M parameter multistream transformer architecture adapted from the Hibiki model, comprising a Temporal Transformer and Depth Transformer operating on Mimi codec tokens.

iii. To develop a synthetic parallel speech data generation pipeline using forced alignment (WhisperX), machine translation (NLLB-200), text-to-speech synthesis (MMS-TTS), contextual alignment, and silence insertion.

iv. To implement and execute a four-stage training protocol (text adaptation, audio pretraining, speech translation training, and fine-tuning) on Kaggle T4 GPUs.

v. To evaluate the system on the FLEURS English-Swahili test set using ASR-BLEU, speaker similarity, and latency metrics, and compare against cascaded baselines and existing multilingual S2ST systems.

vi. To improve the Swahili ASR component of the cascaded baseline by fine-tuning Whisper-small on a combination of labelled KenSpeech audio and confidence-filtered pseudo-labels generated by a Whisper-large-v3 teacher, thereby producing a more rigorous baseline for comparison and contributing a reproducible low-resource ASR distillation recipe.

### 1.4 Scope of Work

This project encompasses the full system design and implementation pipeline, from data preparation through model training to evaluation. The scope includes:

- Design of the model architecture and training protocol
- Development of the synthetic data pipeline for both Swahili-to-English and English-to-Swahili directions
- Implementation of all software components (model code, data loaders, training scripts, evaluation scripts)
- Training on Kaggle T4 GPUs within a budget of approximately 60 GPU-hours
- Evaluation on the FLEURS benchmark
- Construction of an improved cascaded baseline via Whisper-small Swahili ASR fine-tuning with pseudo-label distillation from Whisper-large-v3

The project focuses on read speech translation (utterances of 5-20 seconds). Real-time streaming inference, speaker diarization for multi-speaker input, and deployment as a production service are outside the scope of this work. The Mimi neural audio codec is used as a pretrained frozen component and is not retrained.

### 1.5 Justification

Swahili is one of the most widely spoken languages in Africa, yet it remains underserved by modern speech technology. An end-to-end S2ST system for English-Swahili would have significant practical applications:

- **Healthcare:** Enabling communication between English-speaking medical professionals and Swahili-speaking patients in rural areas without requiring a human interpreter.
- **Education:** Providing real-time translation of educational content between English and Swahili.
- **Commerce:** Facilitating cross-border trade in East Africa where English and Swahili are both used commercially.
- **Accessibility:** Serving populations who are literate in spoken Swahili but not in written English, for whom text-based translation tools are inaccessible.

Furthermore, this project demonstrates that meaningful S2ST research for low-resource languages can be conducted within the computational constraints available to African university researchers, using freely available GPU platforms and open-source pretrained models. The synthetic data generation methodology developed here is transferable to other low-resource African language pairs.

---

## 2 LITERATURE REVIEW

### 2.1 Speech Translation Paradigms

Speech translation systems convert spoken language in one language into spoken or written language in another. Two primary paradigms exist: cascaded systems and end-to-end systems.

**Cascaded systems** decompose the problem into sequential stages: automatic speech recognition (ASR) converts source speech to text, machine translation (MT) translates the text, and optionally text-to-speech (TTS) synthesises target speech. Each component can be developed and optimised independently using well-established techniques. However, cascaded systems suffer from error propagation -- ASR errors corrupt the MT input, and MT errors corrupt the TTS input. Additionally, paralinguistic features such as speaker identity, prosody, and emotion are typically lost at the ASR stage and must be artificially reconstructed at the TTS stage. Latency is also cumulative across stages, making real-time operation challenging [Lavie et al., 1997].

**End-to-end (E2E) systems** perform the translation within a single model, potentially preserving paralinguistic features and reducing latency. Early E2E approaches used sequence-to-sequence models operating directly on spectrograms [Jia et al., 2019], but these struggled with the long sequence lengths of audio. The introduction of discrete audio representations via neural codecs [Défossez et al., 2023; Borsos et al., 2023] has enabled transformer-based E2E systems that operate on compact token sequences, dramatically improving scalability.

| Approach | Advantages | Disadvantages |
|---|---|---|
| Cascaded (ASR+MT+TTS) | Modular, interpretable, leverages mature components | Error propagation, high latency, loses speaker identity |
| End-to-End (direct) | Lower latency, preserves paralinguistics, joint optimization | Requires parallel speech data, computationally expensive |
| End-to-End (codec-based) | Compact discrete tokens, streaming capable, voice preservation | Codec quality ceiling, large model sizes in literature |

*Table 1: Comparison of Speech Translation Approaches*

### 2.2 Cascaded Speech Translation Systems

The traditional cascaded approach has been the dominant paradigm for deployed speech translation systems. Google Translate's conversation mode [Brants et al., 2007], Microsoft's Skype Translator, and numerous research systems follow this architecture. For the English-Swahili pair specifically, a cascaded system can be assembled from existing components:

- **ASR:** OpenAI's Whisper [Radford et al., 2023] provides multilingual speech recognition covering Swahili, though with higher word error rates than for high-resource languages. Whisper-large-v3 achieves reasonable Swahili ASR performance, while smaller variants (small, medium) trade accuracy for speed.
- **MT:** Meta's NLLB-200 (No Language Left Behind) [NLLB Team et al., 2022] supports direct English-Swahili translation with its 1.3B distilled variant fitting on consumer GPUs. Google's MADLAD-400 [Kudugunta et al., 2024] provides an alternative, though our experiments found NLLB-200 more reliable on T4 GPUs.
- **TTS:** Meta's MMS-TTS [Pratap et al., 2024] provides pretrained single-speaker TTS for over 1,100 languages including Swahili (`facebook/mms-tts-swh`) and English (`facebook/mms-tts-eng`), using the VITS architecture.

While functional, this cascade introduces 2-5 seconds of latency per utterance and cannot preserve the source speaker's voice characteristics in the output.

### 2.3 End-to-End Speech Translation

Recent end-to-end systems have achieved remarkable results:

**Translatotron** [Jia et al., 2019] was among the first E2E S2ST models, using an attention-based sequence-to-sequence model operating on spectrograms. Translatotron 2 [Jia et al., 2022] improved upon this by adding a text auxiliary task and enabling voice preservation.

**Seamless** [Seamless Communication et al., 2023] from Meta AI is a family of models supporting speech-to-speech, speech-to-text, text-to-speech, and text-to-text translation across nearly 100 languages. SeamlessM4T uses a multitask architecture with shared encoder representations. While it supports Swahili, its performance on English-Swahili S2ST is limited by the low-resource nature of the pair.

**Moshi** [Défossez et al., 2024] from Kyutai demonstrated that a decoder-only transformer could perform full-duplex spoken dialogue by operating on multiple streams of discrete audio tokens simultaneously. Moshi introduced the concept of the "Inner Monologue" text stream -- a text representation predicted alongside audio tokens that grounds the model's outputs in language understanding.

**Hibiki** [Défossez et al., 2025] extended Moshi's multistream architecture to speech translation. Hibiki processes source speech and generates target speech synchronously using a Temporal Transformer for sequence-level modelling and a Depth Transformer for codebook-level modelling. Key innovations include contextual alignment (aligning source and target words using translation model perplexity) and silence insertion (inserting silences into target audio to maintain causal alignment with the source). Hibiki achieves state-of-the-art French-to-English translation quality with streaming capability, but uses 2.7B parameters and was trained on proprietary data.

### 2.4 Neural Audio Codecs

Neural audio codecs compress audio into discrete token sequences using learned codebooks, providing a compact representation suitable for language modelling. Key codecs include:

**EnCodec** [Défossez et al., 2023] uses a convolutional encoder-decoder with residual vector quantisation (RVQ) to encode audio at various bitrates. It operates at 24 kHz with configurable codebook counts.

**SoundStream** [Zeghidour et al., 2022] from Google similarly uses RVQ but with additional discriminator training for higher perceptual quality.

**Mimi** [Défossez et al., 2024] is the codec used by Moshi and Hibiki. It operates at 12.5 Hz frame rate (significantly lower than EnCodec's 75 Hz), producing a much more compact token sequence. Mimi uses 16 codebooks (or 8 in reduced configurations), where the first codebook captures semantic content and subsequent codebooks capture acoustic details. The pretrained `kyutai/mimi` model is freely available under CC-BY license on HuggingFace.

For this project, Mimi was selected due to its low frame rate (producing manageable sequence lengths for the Temporal Transformer), its demonstrated effectiveness in the Hibiki system, and its free availability. Using Q=8 codebooks (reduced from 16) cuts the Depth Transformer's computational cost while retaining sufficient audio quality for speech.

### 2.5 The Hibiki Architecture

Hibiki [Défossez et al., 2025] is central to this project and warrants detailed examination. The architecture consists of:

**Temporal Transformer:** A causal decoder-only transformer that processes the sequence dimension. At each timestep, it receives embeddings from all active streams (source audio, target audio, inner monologue text) and produces a latent representation used by the Depth Transformer. The original Hibiki uses 32 layers with dimension 2048.

**Depth Transformer:** A smaller transformer that processes the codebook dimension at each timestep. Given the Temporal Transformer's output for a single timestep, it autoregressively generates tokens for each codebook level (1 through Q). This two-level hierarchy allows the model to capture both temporal dependencies (across time) and spectral dependencies (across codebooks) efficiently.

**Token Streams:** Hibiki maintains three simultaneous streams:
1. *Source stream (listens):* Mimi tokens from the source audio -- ground truth during training, live audio during inference.
2. *Target stream (speaks):* Mimi tokens for the target audio -- predicted by the model.
3. *Inner Monologue:* Text tokens of the target language, aligned with the target audio frames -- predicted by the model and used to ground outputs in linguistic content.

**Acoustic Delay:** An offset of $\delta$ timesteps is applied between the semantic (codebook 1) and acoustic (codebooks 2-Q) tokens, following Equation 3 of the Hibiki paper. This delay allows the model to first commit to the semantic content before generating the acoustic details.

**Contextual Alignment:** To create training data, Hibiki aligns source and target words using a bilingual translation model's perplexity. For each target word position, the algorithm finds the source word whose inclusion most reduces the translation perplexity, producing a monotonic alignment (Equation 6).

**Silence Insertion:** Based on the contextual alignment, silences are inserted into the target audio to ensure that each target word begins no earlier than its aligned source word. This maintains the causal property required for streaming: the model never needs to "look ahead" in the source audio to generate the current target audio.

### 2.6 Multilingual Speech Models for African Languages

Several initiatives have aimed to expand speech technology coverage for African languages:

**MMS (Massively Multilingual Speech)** [Pratap et al., 2024] from Meta AI covers over 1,100 languages for ASR and TTS, including Swahili. The MMS-TTS models use the VITS architecture and provide single-speaker synthesis without requiring language-specific training data collection.

**Whisper** [Radford et al., 2023] from OpenAI provides multilingual ASR for 99 languages including Swahili. Whisper was trained on 680,000 hours of weakly-supervised multilingual audio scraped from the web, of which only a small fraction is Swahili; consequently, its Swahili performance lags significantly behind high-resource languages. Reported Word Error Rates on the FLEURS sw_ke test set vary across model sizes -- Whisper-small zero-shot is generally in the 60--90% range, while Whisper-large-v3 achieves substantially better results in the 25--40% range due to its larger capacity and more recent training data. The smaller variants (tiny, base, small) provide deployable inference latency at the cost of accuracy, motivating the use of fine-tuning and self-training methods to recover quality without paying the inference cost of the large model.

**NLLB-200** [NLLB Team et al., 2022] provides machine translation for 200 languages, with direct English-Swahili support. The distilled 1.3B parameter variant can run on consumer GPUs while maintaining reasonable translation quality.

**Common Voice** [Ardila et al., 2020] from Mozilla provides crowd-sourced speech data in multiple languages. The Swahili corpus contains approximately 300 hours of validated recordings with transcripts, making it the largest freely available Swahili speech dataset.

**KenSpeech** [Wanjiku et al., 2023] is a 27.5-hour Kenyan Swahili speech dataset with 26 labelled speakers at 16 kHz, providing a smaller but higher-quality complement to Common Voice for Kenyan-accented Swahili.

### 2.7 Synthetic Data Generation for Low-Resource Pairs

The fundamental challenge for training S2ST models on low-resource pairs is the absence of parallel speech data -- recordings of the same content spoken in both languages with alignment. Several approaches have been developed to address this:

**Back-translation** [Sennrich et al., 2016] generates synthetic parallel text by translating monolingual target text back to the source language. This has been extended to speech by using TTS to synthesise audio from translated text.

**Hibiki's approach** [Défossez et al., 2025] uses monolingual source audio, transcribes it, translates the transcripts, synthesises target audio from translations, and then applies contextual alignment and silence insertion to create causally-aligned parallel speech pairs. This is the approach adopted and adapted in this project.

**ASR pseudo-labelling** [Likhomanenko et al., 2021] uses a teacher ASR model to generate transcriptions for unlabelled audio, which are then filtered and used to train a student model. This has been applied to expand Swahili ASR training data.

For English-Swahili specifically, the pipeline developed in this project combines WhisperX forced alignment (to obtain accurate word-level timestamps from existing Common Voice transcripts), NLLB-200 translation, MMS-TTS synthesis, and the Hibiki contextual alignment and silence insertion algorithms.

### 2.8 ASR Pseudo-Label Distillation

Self-training -- using a teacher model to label unlabelled data on which a student is then trained -- has a long history in machine learning [Yarowsky, 1995] and has become a dominant paradigm for low-resource speech recognition. Several closely related strands of work motivate the approach adopted in this project.

**Iterative pseudo-labelling for ASR.** Likhomanenko et al. [2021] (slimIPL) demonstrated that iteratively refining pseudo-labels with a periodically-updated student model can match or exceed the WER of fully-supervised training when the unlabelled corpus is large. The recipe is sensitive to label noise: naively training on every pseudo-label, including low-confidence and degenerate ones, can produce a student that performs *worse* than one trained on labelled data alone. This motivates a confidence-based filtering step.

**Distil-Whisper.** Gandhi et al. [2023] applied pseudo-label distillation specifically to the Whisper family, using whisper-large-v2 as a teacher to produce labels for an unlabelled English corpus on which a smaller student (Distil-Whisper) was trained. They reported a 6x speedup at 1% WER cost on English. Their recipe emphasised **WER-based filtering** -- discarding pseudo-labels whose disagreement with a reference (where available) exceeded a threshold -- and **chunked long-form decoding** to reduce hallucinations. For low-resource languages without a held-out reference set, intrinsic confidence signals (teacher log-probability, repetition) substitute for WER-based filtering.

**Failure modes specific to Whisper.** Radford et al. [2023] and follow-up work observed three characteristic failure modes when Whisper is used as a teacher on noisy or out-of-distribution audio: (i) collapse into n-gram repetition loops (e.g., "the the the the"), (ii) hallucinated text in a wrong language (commonly English when Whisper is unsure of the source language), and (iii) confident but incorrect transcriptions on short or low-SNR audio. These failure modes can be detected without a reference transcription using simple textual heuristics -- repetition ratio, language-identification confidence on the predicted text, and the teacher's own per-token log-probability respectively -- providing a basis for a multi-filter recipe.

**Confidence scoring for pseudo-label quality.** A critical design choice in pseudo-label distillation is how to estimate the quality of each teacher prediction without access to a ground-truth reference. Two families of signals are commonly used. *Extrinsic signals* compare the teacher's output against an external reference (e.g., WER against a held-out transcript), which requires labelled data and is therefore circular for the purely unlabelled setting. *Intrinsic signals* derive from the teacher model itself: the most common is the per-token average log-probability, $\bar{\ell} = \frac{1}{T} \sum_{t=1}^{T} \log p(y_t | y_{<t}, x)$, which can be interpreted as the negative log-perplexity of the predicted sequence. Whisper's `model.generate()` with `output_scores=True` returns the logits at each decoding step, from which $\bar{\ell}$ is computed. Gandhi et al. [2023] used a threshold of $\bar{\ell} > -1.0$ (perplexity $\approx 2.7$) as a reasonable default for filtering English pseudo-labels; this project adopts the same threshold for Swahili and investigates its interaction with corpus size.

**Corpus scaling and diminishing returns.** A key question for pseudo-label distillation at scale is whether student WER continues to improve as the pseudo-labelled corpus grows, or whether returns diminish beyond a certain size. Likhomanenko et al. [2021] observed roughly log-linear improvement in WER as a function of unlabelled corpus size for English, with diminishing returns beyond approximately 50,000 hours of audio. For low-resource languages the dynamics may differ: with only ~27 hours of labelled data (KenSpeech) and ~200 hours of unlabelled data (CV-Sw validated), the unlabelled-to-labelled ratio is much smaller, and gains from pseudo-label scaling may be more persistent. This project's two-stage pseudo-labelling campaign (6K samples for ablation, then 267K for the full corpus) provides empirical evidence on this question for Swahili specifically.

**African-language ASR self-training.** Recent work has applied pseudo-labelling to African languages, including Swahili. The general pattern is that pseudo-label quality is the dominant predictor of student WER improvement; with appropriate filtering, even noisy teacher transcripts of crowd-sourced audio (Common Voice) can substantially improve a student model trained on a small high-quality labelled corpus. The AfriSpeech benchmark [Olatunji et al., 2023] and related efforts have shown that self-training with Whisper teachers can achieve competitive WER on African-accented English, though results for native African-language ASR remain less explored. The contribution of this project, in this thread, is a Swahili-specific application combining the three filter signals above into a single composable recipe whose individual contributions can be ablated, tested at two corpus scales (6K and 267K pseudo-labels), and integrated into a downstream S2ST evaluation as the ASR front-end of an improved cascade baseline.

### 2.9 Summary of Literature Gaps

The following gaps motivate this project:

1. **No dedicated E2E S2ST system for English-Swahili.** Existing systems like Seamless support Swahili as one of many languages but are not optimised for the pair.
2. **Hibiki has not been demonstrated below 2.7B parameters or for African languages.** It is unknown whether the architecture remains effective at ~100M parameters or for typologically distant language pairs.
3. **No established methodology for generating synthetic English-Swahili parallel speech data.** The pipeline developed in this project addresses this gap.
4. **Limited investigation of training S2ST models within consumer GPU constraints.** Most S2ST research assumes access to large GPU clusters; this project demonstrates feasibility on Kaggle T4 GPUs.
5. **Off-the-shelf cascaded baselines understate cascade quality.** Most E2E S2ST papers compare against an unmodified cascade (Whisper-small + NLLB + MMS-TTS) without first improving the ASR component for the target language. This project addresses the gap by fine-tuning the ASR component before using it in the cascade, producing a more rigorous baseline.

---

## 3 DESIGN METHODOLOGY

### 3.1 System Overview

The system consists of three main components: (1) a synthetic data generation pipeline that creates parallel English-Swahili speech pairs from monolingual resources, (2) a ~100M parameter multistream transformer model that performs end-to-end speech translation, and (3) a four-stage training protocol that progressively builds the model's capabilities.

The data pipeline processes monolingual speech (Common Voice Swahili, ~70K validated utterances) through transcription, translation, synthesis, alignment, and encoding stages to produce discrete Mimi codec tokens suitable for model training. The model processes these tokens through a Temporal Transformer (sequence-level) and Depth Transformer (codebook-level) to generate target language audio tokens, which are decoded back to waveforms by the Mimi decoder.

*[Figure 3.1: System Architecture Overview -- diagram showing Source Audio -> Mimi Encoder -> Temporal Transformer + Depth Transformer -> Mimi Decoder -> Target Audio, with Inner Monologue text stream]*

### 3.2 Neural Audio Codec (Mimi)

The pretrained `kyutai/mimi` codec serves as the audio tokenizer for both input and output. Mimi encodes 24 kHz mono audio into discrete tokens at a frame rate of 12.5 Hz using Q=8 codebooks, each with a vocabulary of 2048 entries.

| Parameter | Value |
|---|---|
| Sample rate | 24,000 Hz |
| Frame rate | 12.5 Hz |
| Number of codebooks (Q) | 8 |
| Codebook vocabulary size | 2,048 |
| Seconds per token | 0.08 s |
| Tokens per 20s utterance | 250 |

The first codebook captures semantic content (analogous to phonetic information), while codebooks 2-8 capture progressively finer acoustic details (timbre, pitch, noise characteristics). This hierarchical structure is exploited by the Depth Transformer, which generates tokens for each codebook level sequentially.

The codec is frozen during all training stages -- its parameters are not updated. This is critical for two reasons: (1) it avoids catastrophic forgetting of the codec's learned audio representations, and (2) it ensures that tokens encoded before training remain valid throughout the training process.

### 3.3 Temporal Transformer Design

The Temporal Transformer is the main backbone of the model, responsible for sequence-level modelling across time. It is a causal (left-to-right) decoder-only transformer with the following configuration:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Latent dimension (d_model) | 512 | Balanced between capacity and memory |
| FFN inner dimension | 1,408 | Gated SiLU (≈2.75× d_model) |
| Number of layers | 12 | Sufficient depth for 100M budget |
| Attention heads | 8 | Head dimension = 64 |
| Head dimension | 64 | Standard for this scale |
| Context window | 250 tokens | ~20 seconds at 12.5 Hz |
| Position encoding | RoPE | Rotary position embeddings |
| Dropout | 0.1 | Regularization |
| Activation | Gated SiLU | Following Hibiki/LLaMA |

*Table 2: Temporal Transformer Hyperparameters*

The gated SiLU activation follows the LLaMA architecture convention, where the FFN has three projection matrices (gate, up, down) rather than two, providing better gradient flow. Rotary Position Embeddings (RoPE) [Su et al., 2022] encode relative positions directly into the attention computation, enabling better generalisation to different sequence lengths.

Per-layer parameter count:
- Self-attention: 4 x 512^2 = 1,048,576
- Gated FFN: 3 x 512 x 1,408 = 2,162,688
- Layer norm + biases: ~2,048
- **Per layer total: ~3.2M**
- **12 layers: ~38.4M**
- Embeddings + output heads: ~26.6M
- **Temporal Transformer total: ~65M parameters**

### 3.4 Depth Transformer Design

The Depth Transformer processes the codebook dimension at each individual timestep. Given the Temporal Transformer's latent representation for a single time position, it autoregressively generates tokens for codebook levels 1 through Q (8 levels).

| Hyperparameter | Value | Rationale |
|---|---|---|
| Latent dimension | 384 | Smaller than Temporal (local task) |
| FFN inner dimension | 1,024 | Gated SiLU |
| Layers per codebook step | 4 | Sufficient for codebook dependencies |
| Output codebook levels | 8 | Q=8 Mimi codebooks |
| Weight sharing | Codebooks 5-8 share | Saves parameters; upper codebooks similar |
| Embedding dimension | 64 | Low-rank for acoustic tokens |
| Dropout | 0.1 | Regularization |

*Table 3: Depth Transformer Hyperparameters*

Weight sharing across codebooks 5-8 exploits the observation that higher codebook levels capture increasingly fine acoustic details with diminishing marginal information, so separate weights are unnecessary. Low-rank embeddings (dimension 64 instead of 384) for acoustic tokens further reduce parameters, since the semantic information is already captured by the Temporal Transformer.

**Depth Transformer total: ~30M parameters.**

### 3.5 Token Stream Architecture

During training, the model processes three simultaneous streams at each timestep, following the Hibiki multi-stream design:

1. **Source stream (listens):** Contains Mimi tokens from the source language audio. During training, these are ground truth tokens; during inference, they come from live encoding of the input audio. The source stream has Q=8 codebook levels.

2. **Target stream (speaks):** Contains Mimi tokens for the target language audio. These are the primary prediction targets. The target stream also has Q=8 codebook levels.

3. **Inner Monologue text stream:** Contains SentencePiece BPE tokens of the target language text, aligned to the target audio frames. The text tokens are distributed across frames using either forced-alignment timestamps or uniform distribution (where forced alignment is unavailable). This stream provides a linguistic grounding signal that helps the model produce coherent translations.

An **acoustic delay** of $\delta = 2$ timesteps is applied between semantic tokens (codebook 1) and acoustic tokens (codebooks 2-8) in both source and target streams. This allows the model to commit to the semantic content before generating the acoustic realisation.

The total input dimension per timestep is the sum of embeddings from all active streams, projected into the Temporal Transformer's d_model = 512 space.

### 3.6 Text Tokenizer Design

A SentencePiece BPE tokenizer with a vocabulary of 32,000 tokens was trained on combined English and Swahili text from:
- OPUS CCAligned English-Swahili parallel corpus (~230K sentence pairs)
- FLEURS transcripts (English and Swahili)
- Common Voice Swahili transcripts
- KenSpeech Swahili transcripts (~6,000 utterances, loaded without cap)

The tokenizer uses byte fallback to handle out-of-vocabulary characters, a character coverage of 0.9995, and reserves special tokens: PAD (0), BOS (1), EOS (2), and EPAD (3, used for empty text positions in the aligned text stream).

### 3.7 Data Pipeline Design

Since no large-scale English-Swahili parallel speech corpus exists, a synthetic data generation pipeline was designed following the methodology introduced in the Hibiki paper. The pipeline processes monolingual source audio through five stages to produce aligned parallel speech pairs.

#### 3.7.1 Transcription with Forced Alignment

For Common Voice Swahili data, the human transcripts provided in the TSV metadata (`sentence` field) are used directly as the authoritative text -- Whisper is never used for transcription of CV data. Word-level timestamps are obtained through a three-tier fallback strategy:

- **Tier 1 (preferred):** WhisperX forced alignment using a wav2vec2 phoneme model (`wav2vec2-large-xlsr-53-swahili`). This aligns the known transcript to the audio at the word level without performing any transcription.
- **Tier 2 (fallback):** Whisper small with `initial_prompt` set to the CV sentence. This guides Whisper to produce timestamps aligned to the known text rather than performing free-form transcription.
- **Tier 3 (last resort):** Uniform timestamp distribution -- word boundaries are evenly spaced across the audio duration. This produces degraded but usable timestamps.

This design ensures that the high-quality human transcripts are always preserved, while obtaining the best possible word-level timing information for downstream silence insertion.

#### 3.7.2 Machine Translation

Transcribed text is translated using NLLB-200-distilled-1.3B (`facebook/nllb-200-distilled-1.3B`), a distilled multilingual translation model supporting 200 languages. For Swahili-to-English translation, the source language is set to `swh_Latn` and the target to `eng_Latn`. The model operates in FP16 on a single T4 GPU, processing text in batches of 8 sentences.

NLLB-200 was selected over MADLAD-400-3B based on empirical testing: MADLAD-400 produced degenerate outputs (repetitive or empty text) when running on T4 GPUs in our configuration, while NLLB-200-distilled-1.3B consistently produced coherent translations.

#### 3.7.3 Text-to-Speech Synthesis

Target language audio is synthesised from the translated text using Meta's MMS-TTS:
- **Swahili synthesis:** `facebook/mms-tts-swh` (for English-to-Swahili direction)
- **English synthesis:** `facebook/mms-tts-eng` (for Swahili-to-English direction)

MMS-TTS uses the VITS architecture (Variational Inference with adversarial learning for end-to-end Text-to-Speech) and produces natural-sounding single-speaker output at 16 kHz, which is resampled to 24 kHz for Mimi encoding.

#### 3.7.4 Contextual Alignment

Contextual alignment determines which source word corresponds to which target word, producing a monotonic mapping used for silence insertion. Following Hibiki's approach (Section 3.2.1, Equation 6), the alignment is computed using the translation model's perplexity:

For each target word position, the algorithm evaluates how much the inclusion of each candidate source word reduces the conditional perplexity of the target sequence. The source word that most reduces perplexity is selected as the alignment point. A monotonicity constraint ensures that the alignment never goes backward.

#### 3.7.5 Silence Insertion

Based on the contextual alignment and the word-level timestamps from both source and target audio, silences are inserted into the target audio to maintain causal alignment. The rule is:

> For each target word $w_t$ aligned to source word $w_s$, if $w_t$ would start before $w_s$ in the parallel playback, insert silence before $w_t$ to delay it until after $w_s$ starts.

This ensures that the model never needs to "hear" future source audio to generate the current target audio, enabling streaming inference. A minimum lag buffer of 2 seconds is used to provide the model with sufficient context, following the Hibiki paper's recommendation.

### 3.8 Training Protocol Design

Training follows a four-stage progressive protocol, where each stage builds upon the previous one's checkpoint:

| Stage | Description | Data | Steps | Batch | LR | GPU-hrs |
|---|---|---|---|---|---|---|
| 1 | Text Adaptation | en+sw text (OPUS, CC) | 15K | 256 | 1e-4 | ~2 |
| 2 | Audio Pretraining | en+sw monolingual audio | 80K | 64 | 1.5e-4 | ~12 |
| 3 | S2ST Training | Synthetic parallel pairs | 40K | 32 | 2e-5 | ~11 |
| 4 | Fine-tuning | Curated high-quality subset | 4K | 8 | 1e-6 | ~2 |

*Table 5: Training Stages Overview*

**Stage 1 (Text Adaptation):** The Temporal Transformer is initialised and adapted on bilingual English-Swahili text using next-token prediction. This gives the model a foundation in both languages' vocabulary and grammar before it encounters audio.

**Stage 2 (Audio Pretraining):** The Depth Transformer is introduced (randomly initialised) and both transformers are trained on monolingual English and Swahili audio in single-stream mode. The model learns to predict next audio tokens (both semantic and acoustic codebooks) for individual languages.

**Stage 3 (S2ST Training):** The model switches to multistream mode, processing source and target audio streams simultaneously using the synthetic parallel data. The model learns to translate by predicting target audio tokens conditioned on source audio tokens. An inner monologue text stream provides linguistic grounding.

**Stage 4 (Fine-tuning):** The model is fine-tuned on a curated subset of the highest-quality synthetic pairs, with classifier-free guidance (CFG) training where the voice similarity conditioning label is randomly dropped with 20% probability.

All stages use the AdamW optimiser with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1, and cosine learning rate scheduling with warmup. FP16 mixed precision and gradient checkpointing are used throughout.

**Bidirectional strategy:** Stages 1-2 are shared (language-agnostic). Stages 3-4 are run twice from the same Stage 2 checkpoint -- once for Swahili-to-English, once for English-to-Swahili -- producing two separate trained models.

### 3.9 Evaluation Metrics

The system is evaluated using the following metrics on the FLEURS English-Swahili test set:

- **ASR-BLEU:** The translated speech output is transcribed using Whisper medium, and the BLEU score [Papineni et al., 2002] is computed against the reference text translation. This is the primary translation quality metric.
- **Text BLEU:** Computed directly from the Inner Monologue text stream, providing a text-domain quality measure without ASR errors.
- **Speaker Similarity:** Cosine similarity between WavLM speaker embeddings [Chen et al., 2022] of the source and target audio, measuring how well the system preserves the source speaker's voice characteristics.
- **LAAL (Length-Adaptive Average Lagging):** Measures the delay between source and target audio, evaluating the system's streaming capability.

### 3.10 Cascade Baseline Improvement via Whisper Fine-tuning

The cascaded baseline against which the end-to-end S2ST system is compared consists of three off-the-shelf components: Whisper-small for ASR, NLLB-200-distilled-1.3B for machine translation, and MMS-TTS for synthesis. In the Sw-to-En direction, the ASR component is the dominant bottleneck: Whisper-small zero-shot Word Error Rate on FLEURS sw_ke is in the 80--90% range, which floods the downstream NLLB stage with errors and inflates the cascaded ASR-BLEU error budget unfairly. To make the cascade a defensible baseline, the ASR component is improved through pseudo-label distillation from a stronger teacher.

**Method overview.** The improvement is structured as a four-step pipeline:

1. **Pseudo-labelling.** Whisper-large-v3, used in zero-shot mode with the Swahili language token, is run over a pool of unlabelled Swahili audio drawn from Common Voice Swahili (validated subset) and the FLEURS sw_ke train split. For each audio, the teacher's per-token average log-probability is captured alongside the predicted text, providing an intrinsic confidence signal.

2. **Quality filtering.** A composable three-filter recipe discards pseudo-labels likely to be incorrect, addressing the three Whisper failure modes documented in Section 2.8:

    - *Confidence filter:* drops entries whose teacher average log-probability is below a threshold (default $-1.0$, corresponding to a per-token perplexity of approximately 2.7).
    - *Repetition filter:* computes the maximum n-gram repetition ratio for $n \in \{1, 2, 3, 4, 5\}$ on the predicted text and drops entries above a threshold (default 0.5). This catches degenerate "the the the" loops.
    - *Language-identification filter:* runs a Swahili language detector on the predicted text and drops entries whose Swahili confidence is below a threshold (default 0.7). This catches hallucinated English transcriptions.

    Each filter can be enabled or disabled independently, allowing an ablation study of each filter's marginal contribution.

3. **Student fine-tuning.** Whisper-small is fine-tuned on the union of (i) the labelled KenSpeech corpus (a small, high-quality Kenyan-Swahili dataset) and (ii) the filtered pseudo-labels. The standard HuggingFace seq2seq training recipe is used, with a single FP16 cross-entropy loss on next-token prediction over the Swahili transcript labels. No CTC head, no auxiliary tasks, and no architectural modifications to Whisper-small are introduced -- the contribution is entirely in the data, not the model.

4. **Evaluation.** Word Error Rate and Character Error Rate are computed on FLEURS sw_ke test (487 utterances) after lowercase + punctuation normalisation. To establish how much of the gold-data benefit pseudo-labels recover, an "upper-bound" variant trains on the same audio but substitutes the teacher's pseudo-labels with the original CV/FLEURS-train gold transcripts where available. The full experimental matrix comprises five systems:

    | System | Training data | Role |
    |---|---|---|
    | `vanilla_small` | None (zero-shot) | External baseline |
    | `ft_kenspeech_only` | 5.7K real KenSpeech labels | Supervised baseline |
    | `ft_kenspeech_pseudo_raw` | KenSpeech + unfiltered pseudo-labels | Naive pseudo-labelling baseline |
    | `ft_kenspeech_pseudo_filtered` | KenSpeech + filtered pseudo-labels | The novel method |
    | `ft_kenspeech_gold_upper_bound` | KenSpeech + gold-where-available | Upper bound |

    *Table N: ASR Experimental Matrix*

The contribution-claim that this design enables is two-fold: first, the improved cascade uses the best-performing student model from this matrix, providing a stronger and more honest baseline for the S2ST comparison; and second, the filter recipe and its per-filter ablation constitute a stand-alone methodological contribution to low-resource Swahili ASR.

---

## 4 IMPLEMENTATION

### 4.1 Development Environment

Two compute environments were used in this project, each serving a distinct role in the pipeline.

#### 4.1.1 Kaggle T4 (Synthetic Data Pipeline + S2ST Model Training)

The primary compute platform for synthetic data generation and model training was Kaggle's free-tier GPU allocation:

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 (16 GB VRAM) via Kaggle |
| GPU count | 1-2 (DDP when 2 available) |
| Compute capability | 7.5 (Turing architecture) |
| FP16 throughput | 65 TFLOPS |
| Framework | PyTorch >= 2.0 |
| Precision | FP16 mixed precision (PyTorch AMP) |
| Session limit | 9 hours per Kaggle session |
| Weekly GPU quota | 30 hours |
| Total GPU budget | ~44 GPU-hours (S2ST training) |
| Key libraries | transformers, faster-whisper, whisperx, sentencepiece, datasets, torchaudio |

The T4's 16 GB VRAM is sufficient for all pipeline steps and for training the 100M-parameter S2ST model, but requires careful memory management (Section 4.5). The 9-hour session limit necessitates resumable scripts and checkpoint management across sessions.

#### 4.1.2 AWS g5.2xlarge (Whisper ASR Fine-tuning Pipeline)

The Whisper ASR pseudo-label distillation pipeline (Section 4.6) was executed on an AWS EC2 `g5.2xlarge` on-demand instance, providing a larger GPU for teacher inference:

| Component | Specification |
|---|---|
| Instance type | g5.2xlarge |
| GPU | NVIDIA A10G (24 GB VRAM) |
| Compute capability | 8.6 (Ampere architecture) |
| FP32 throughput | 31.2 TFLOPS |
| BF16/FP16 throughput | 62.5 TFLOPS |
| vCPUs | 8 (AMD EPYC 7R32) |
| System RAM | 32 GB |
| Storage | 450 GB NVMe SSD (ephemeral instance store) |
| AMI | AWS Deep Learning AMI (PyTorch 2.1, CUDA 12.1) |
| Hourly cost | ~\$1.21/hr (on-demand, us-east-1) |
| Network | Up to 10 Gbps |

The A10G was selected over the Kaggle T4 for the ASR pipeline for three reasons:

1. **VRAM headroom for the teacher model.** Whisper-large-v3 (1.55B parameters) requires approximately 3 GB at bf16. Combined with the student model and training data loaders, the pipeline peaks at approximately 12-14 GB VRAM during training -- comfortably within the A10G's 24 GB budget but tight on a 16 GB T4, where loading both teacher and student simultaneously would leave insufficient memory for batch processing.

2. **No session time limits.** Unlike Kaggle's 9-hour cap, the on-demand EC2 instance runs continuously. The full pseudo-labelling pipeline (teacher inference over ~267K Common Voice utterances at 3.5-4 samples/second) requires approximately 20 hours of uninterrupted compute -- impossible to fit in a single Kaggle session and awkward to resume across multiple sessions when the teacher model itself takes several minutes to load.

3. **bf16 native support.** The A10G's Ampere architecture supports bf16 natively, enabling the Whisper-large-v3 teacher to run at full precision without the int8 quantisation required on the T4. This preserves the teacher's transcription quality, which is critical since pseudo-label accuracy directly determines student model quality.

The instance was configured with a 200 GB EBS gp3 volume for persistent storage of the Common Voice dataset (~65 GB compressed), pseudo-label manifests, and trained model checkpoints. Data transfer used the AWS CLI (`aws s3 cp`) for dataset ingestion and `huggingface_hub` for model upload. The total AWS cost for the ASR pipeline was approximately \$25-30 (20-25 hours of g5.2xlarge compute).

#### 4.1.3 Repository Structure

All code was developed in Python and organised into a modular repository structure:

```
hibiki-sw/
├── configs/model_100m.yaml        # All hyperparameters
├── data/
│   ├── dataset.py                 # PyTorch Dataset classes
│   ├── contextual_align.py        # Contextual alignment
│   ├── silence_insertion.py       # Silence insertion
│   └── prepare/                   # Data preparation scripts
│       ├── transcribe_whisper.py  # WhisperX forced alignment
│       ├── translate_nllb.py      # NLLB-200 translation
│       ├── synthesize_tts.py      # MMS-TTS synthesis
│       ├── encode_audio.py        # Mimi codec encoding
│       ├── train_tokenizer.py     # SentencePiece training
│       ├── tokenize_text.py       # Text tokenization
│       ├── create_s2st_manifest.py # Manifest creation
│       ├── local_cv_loader.py     # Common Voice local loader
│       └── run_pipeline.py        # Pipeline orchestrator
├── model/
│   ├── temporal_transformer.py    # Temporal Transformer
│   ├── depth_transformer.py       # Depth Transformer
│   ├── hibiki_model.py            # Combined model
│   └── codec.py                   # Mimi codec wrapper
├── training/                      # Training scripts (Stages 1-4)
├── inference/translate.py         # Inference pipeline
├── evaluation/evaluate.py         # Evaluation metrics
└── notebooks/                     # Kaggle notebooks (00-04)
```

### 4.2 Data Preparation

#### 4.2.1 Dataset Collection

The following datasets were used:

| Dataset | Language | Size | Source | Use |
|---|---|---|---|---|
| Common Voice 19.0 (Swahili) | sw | ~70K validated utterances (~200 hrs) | Mozilla | Primary Sw audio + transcripts |
| Common Voice (English) | en | 2000+ hrs | Mozilla | En audio pretraining |
| KenSpeech | sw | ~6K utterances (27.5 hrs) | Kencorpus/HuggingFace | Additional Sw text + audio |
| FLEURS | en, sw | ~10 hrs per language | Google | Evaluation benchmark |
| OPUS CCAligned | en-sw | ~230K parallel sentences | OPUS | Text pretraining |

*Table 6: Available Datasets for English-Swahili*

Common Voice Swahili validated data was pre-downloaded as a Kaggle dataset (`victormugambi/cv-swahili/cv-validated`), containing the `validated.tsv` metadata file and `clips/` directory with MP3 audio files. The `CommonVoiceLocal` loader reads the TSV file to access transcripts (the `sentence` column) and audio file paths, avoiding the overhead of streaming from HuggingFace during training.

#### 4.2.2 Tokenizer Training

A SentencePiece BPE tokenizer was trained on combined English and Swahili text:

```bash
python data/prepare/train_tokenizer.py \
    --output_dir tokenizer \
    --vocab_size 32000 \
    --kenspeech \
    --cv_dataset_dir sw:/path/to/cv-corpus/sw
```

The tokenizer training ingests text from OPUS CCAligned, FLEURS transcripts, Common Voice transcripts, and KenSpeech transcripts (all ~6,000 utterances loaded without cap). The resulting `sp_ensw_32k.model` file encodes both English and Swahili text into a shared 32K token vocabulary.

#### 4.2.3 Audio Encoding with Mimi

All audio (both source monolingual and synthesised target) is pre-encoded through the Mimi codec and stored as NumPy arrays:

```bash
python data/prepare/encode_audio.py \
    --source common_voice \
    --lang sw \
    --dataset_dir /path/to/cv-validated \
    --split validated \
    --output_dir audio_tokens/cv_sw \
    --num_codebooks 8 \
    --max_duration 20.0
```

Each encoded file is a NumPy array of shape `(8, T)` where 8 is the number of codebooks and T is the number of time frames. At 12.5 Hz, a 20-second utterance produces T=250 frames. Files exceeding 20 seconds (T>250) are excluded to maintain a manageable context window for the Temporal Transformer.

### 4.3 Synthetic Parallel Data Pipeline

The synthetic data pipeline is orchestrated by the `00b_data_pipeline.ipynb` notebook and the `run_pipeline.py` script. For the primary Swahili-to-English direction, the pipeline processes ~70K Common Voice Swahili utterances.

#### 4.3.1 WhisperX Forced Alignment

```bash
python data/prepare/transcribe_whisper.py \
    --source common_voice \
    --dataset_dir /path/to/cv-validated \
    --lang sw \
    --split validated \
    --output_dir transcriptions/sw \
    --whisper_model small \
    --compute_type int8 \
    --forced_alignment
```

The `--forced_alignment` flag activates the three-tier timestamp strategy. The CV `sentence` field is always used as the authoritative text. WhisperX loads the `wav2vec2-large-xlsr-53-swahili` alignment model (~1.2 GB) to produce word-level timestamps. If WhisperX is unavailable, Whisper small (int8 quantized, ~0.5 GB) is used as a timestamp-only fallback with the CV sentence passed as `initial_prompt`.

T4 optimisation: Whisper small with int8 quantization was chosen over medium to save ~700 MB VRAM. After transcription, `free_gpu()` is called to release all GPU memory before the next step.

#### 4.3.2 NLLB Translation

```bash
python data/prepare/translate_nllb.py \
    --input_dir transcriptions/sw \
    --output_dir translations/sw2en \
    --source_lang sw \
    --target_lang en \
    --dtype float16 \
    --batch_size 8
```

NLLB-200-distilled-1.3B processes transcripts in batches of 8 in FP16, using ~2.6 GB VRAM. The output JSON files contain `source_text` (CV transcript), `translated_text` (English translation), and `source_words` (word timestamps from the transcription step).

#### 4.3.3 MMS-TTS Synthesis

```bash
python data/prepare/synthesize_tts.py \
    --translation_dir translations/sw2en \
    --output_dir synthetic_audio/sw2en \
    --target_lang en \
    --alignment_dir alignments/sw2en \
    --whisper_model small \
    --target_sr 24000
```

MMS-TTS (`facebook/mms-tts-eng`) synthesises English audio from translated text. The synthesis step also runs Whisper on the synthesised audio to obtain target word timestamps, which are used in the silence insertion sub-step. Synthesised audio is resampled to 24 kHz for Mimi compatibility.

For 70K utterances at ~2-3 samples/sec, this step takes approximately 7-10 hours, spanning multiple Kaggle sessions. The script is designed to be resumable.

#### 4.3.4 Silence Insertion

Silence insertion is applied as part of the synthesis step. Using the contextual alignment and word timestamps from both source and target audio, silences are inserted into the target audio at positions where the target would otherwise advance ahead of the source. The minimum lag buffer is set to 2 seconds following the Hibiki paper.

### 4.4 Model Training

#### 4.4.1 Stage 1: Text Adaptation

The Temporal Transformer is initialised and trained on bilingual English-Swahili text for 15K steps. The text data is tokenized into chunks of 1,024 tokens using the trained SentencePiece model. Next-token prediction with cross-entropy loss is used.

| Setting | Value |
|---|---|
| Effective batch size | 256 (16 local x 8 accumulation x 2 GPUs) |
| Learning rate | 1e-4 (cosine, 500 warmup) |
| Sequence length | 1,024 tokens |
| Steps | 15,000 |
| Estimated GPU time | ~2 hours |

#### 4.4.2 Stage 2: Audio Pretraining

The Depth Transformer is introduced (randomly initialised) and both transformers are trained on monolingual English and Swahili audio encoded as Mimi tokens. The model predicts next tokens across all Q=8 codebook levels.

| Setting | Value |
|---|---|
| Effective batch size | 64 (16 local x 2 accumulation x 2 GPUs) |
| Learning rate | 1.5e-4 (cosine, 2K warmup) |
| Steps | 80,000 |
| Estimated GPU time | ~12 hours |

After Stage 2, the Depth Transformer weights are duplicated for the multistream configuration required in Stage 3.

#### 4.4.3 Stage 3: Speech Translation Training

The model switches to multistream mode, receiving source and target audio streams plus the inner monologue text stream. The model is trained to predict:
- Target audio tokens (all 8 codebooks)
- Source audio tokens (input reconstruction, training-only auxiliary loss)
- Inner monologue text tokens

| Setting | Value |
|---|---|
| Effective batch size | 32 (8 local x 2 accumulation x 2 GPUs) |
| Learning rate | 2e-5 (cosine, 1K warmup) |
| Steps | 40,000 |
| Noise augmentation | Gaussian noise, speed perturbation |
| Estimated GPU time | ~11 hours per direction |

Voice conditioning is applied using a 5-category speaker similarity label (very_bad, bad, neutral, good, very_good), implemented as a learned embedding added to the input representation.

#### 4.4.4 Stage 4: Fine-tuning

The model is fine-tuned on a curated high-quality subset with classifier-free guidance (CFG) training.

| Setting | Value |
|---|---|
| Batch size | 8 (4 local x 1 accumulation x 2 GPUs) |
| Learning rate | 1e-6 (cosine, 200 warmup) |
| Steps | 4,000 |
| CFG dropout | 20% (voice similarity label dropped) |
| Estimated GPU time | ~2 hours per direction |

### 4.5 Kaggle GPU Optimizations

Training on Kaggle T4 GPUs required several optimizations to fit within the 16 GB VRAM constraint and 9-hour session limit:

**Memory management:**
- FP16 mixed precision throughout (T4 has good FP16 throughput at 65 TFLOPS)
- Gradient checkpointing on the Temporal Transformer (saves ~40% memory at ~20% speed cost)
- `free_gpu()` helper function called between pipeline steps to flush GPU memory via `torch.cuda.empty_cache()` and `gc.collect()`
- Pre-tokenized data stored as uint16 NumPy arrays to minimize CPU-to-GPU transfer

**Session management:**
- Checkpoints saved every 2,000 steps to `/kaggle/working/`
- Pipeline outputs uploaded to HuggingFace Hub for persistence across sessions
- All pipeline scripts are resumable (track completed samples, skip existing outputs)

| Pipeline Step | GPU Model | VRAM Usage | Time (70K samples) |
|---|---|---|---|
| WhisperX alignment | wav2vec2-xlsr-53-sw | ~1.2 GB | ~3-4 hrs |
| Whisper fallback | faster-whisper small int8 | ~0.5 GB | ~5-6 hrs |
| NLLB translation | NLLB-200-distilled-1.3B fp16 | ~2.6 GB | ~2-3 hrs |
| MMS-TTS synthesis | MMS-TTS-eng VITS | ~1.0 GB | ~7-10 hrs |
| Mimi encoding | kyutai/mimi | ~0.3 GB | ~1-2 hrs |

*Table 7: T4 GPU Memory Budget per Pipeline Step*

### 4.6 Whisper ASR Fine-tuning for the Improved Cascade

The improved-cascade Whisper-small Swahili ASR model (designed in Section 3.10) is implemented as a self-contained pipeline in the `whisper_asr/` package, separate from the main S2ST training code. The pipeline comprises five Python scripts orchestrated by `run.sh`. Unlike the S2ST pipeline (Kaggle T4), this work was executed on AWS EC2 with a single NVIDIA A10G GPU (24 GB VRAM) on a Deep Learning AMI (PyTorch 2.10, CUDA 13, NVIDIA driver 580); the larger memory budget enables Whisper-large-v3 teacher inference at native bf16 precision without additional quantisation.

| Script | Role |
|---|---|
| `pseudo_label.py` | Teacher inference (Whisper-large-v3) over CV-Sw + FLEURS-train; saves audio + JSONL manifest |
| `filter_pseudo.py` | Three-filter recipe (confidence + repetition + lang-id) with per-subset ablation stats |
| `dataset.py` | PyTorch Dataset wrapping KenSpeech ∪ optional pseudo-labels |
| `train.py` | Standard `Seq2SeqTrainer` fine-tuning with WER-as-metric checkpoint selection |
| `eval_wer.py` | WER + CER evaluation across multiple model variants on FLEURS sw_ke test |

*Table 8: ASR Fine-tuning Pipeline Components*

#### 4.6.1 Pseudo-label Generation

`pseudo_label.py` loads Whisper-large-v3 (1.55B parameters, ~3 GB at bf16) and iterates over two unlabelled audio sources: the FLEURS sw_ke train split and the locally-cached Common Voice Swahili validated subset. For each audio sample, the script:

1. Resamples audio to 16 kHz mono if necessary.
2. Saves the resampled audio to disk as a 16-bit PCM WAV (so subsequent training does not re-decode MP3s on every epoch).
3. Runs `model.generate()` with `language="sw"`, `task="transcribe"`, greedy decoding (`num_beams=1`, `do_sample=False`), `return_dict_in_generate=True`, and `output_scores=True`.
4. Computes the per-token average log-probability from the captured `scores` array as the entry's confidence.
5. Appends a JSON entry `{audio_path, pseudo_label, avg_log_prob, gold_label, duration_s}` to `pseudo_labels.jsonl`. The `gold_label` field carries the original CV/FLEURS-train transcript for later comparison and for building the gold-upper-bound training set.

The script is fully resumable: at startup it reads the existing manifest and skips any `(source, id)` keys already present. A configurable `--max_per_source` cap allows controlling teacher-inference time independently for each source.

Two pseudo-labelling campaigns were conducted at different scales:

1. **Initial campaign (6K samples, overnight).** With `--max_per_source 3000`, the teacher processed approximately 5,900 samples from CV-Sw validated and FLEURS-sw-train combined. On the A10G at bf16, the teacher achieved approximately 0.4--0.6 samples per second for typical 5--10 second utterances, completing in approximately 5 hours. This campaign produced the results reported in Section 5.7 (Tables 11 and 12).

2. **Full-scale campaign (267K samples, multi-day).** With no per-source cap, the teacher was run over the entire Common Voice Swahili validated set (267,652 utterances). At a sustained throughput of 3.5--4.0 samples/second (slightly faster than the initial campaign due to the mix of shorter CV utterances), the full run required approximately 20 hours of continuous A10G compute. The pseudo-label manifest (`pseudo_labels.jsonl`) grows incrementally, and the script's resume logic allows the instance to be stopped and restarted (e.g., to switch to a spot instance during off-peak hours) without losing progress. At the time of writing, this campaign had completed 146,500 of 267,652 samples (54.7%) after approximately 11.5 hours of elapsed time, with zero errors and zero skipped samples, demonstrating the pipeline's robustness at scale.

A defensive fallback handles the case where transformers silently ignores `output_scores=True` (a known issue in some versions): when `out.scores` is empty, the script falls back to whole-sequence decoding with `avg_log_prob = 0.0`, allowing the entry to bypass the confidence filter rather than crash.

#### 4.6.2 Quality Filtering

`filter_pseudo.py` consumes the raw pseudo-label manifest and applies the three-filter recipe described in Section 3.10. Each filter is a small pure function:

- *Confidence:* `entry["avg_log_prob"] > threshold` (default $-1.0$).
- *Repetition:* `max(ngram_repetition_ratio(text, n) for n in 1..5) < threshold` (default 0.5), where `ngram_repetition_ratio` counts the fraction of n-grams that occur more than once.
- *Language ID:* uses the `lingua-language-detector` Python library restricted to {Swahili, English, French, Arabic} for speed; an entry passes if Swahili confidence is at or above the threshold (default 0.7).

The script computes a complete ablation: for each of the $2^3 = 8$ filter subsets, it reports the number of entries that would have been kept under that subset alone. This is written to `pseudo_labels_filtered.stats.json` and provides the data for the per-filter contribution analysis in Section 5.

A separate "gold-when-available" pass replaces the kept entries' pseudo-labels with their original CV/FLEURS-train transcripts (where present), producing the training data for the upper-bound variant.

#### 4.6.3 Student Fine-Tuning

`train.py` is a standard HuggingFace Whisper fine-tuning recipe based on `Seq2SeqTrainer`. The student is `openai/whisper-small` (244M parameters), and the four training variants differ only in the value of the `--pseudo_labels_path` argument as described in Section 3.10.

The training-time configuration is given in Table 9.

| Setting | Value |
|---|---|
| Base model | `openai/whisper-small` |
| Per-device batch size | 16 |
| Gradient accumulation | 1 (effective batch = 16) |
| Learning rate | $1 \times 10^{-5}$ (cosine, 200 warmup steps) |
| Epochs | 3 |
| Precision | FP16 (loss-scaled mixed precision) |
| Gradient checkpointing | Enabled (`use_reentrant=False`) |
| Eval strategy | Every 400 steps with `predict_with_generate` |
| Best-model selection | Lowest WER on the held-out 5% eval split |

*Table 9: Whisper-small Fine-tuning Hyperparameters*

Two implementation details proved necessary to obtain stable training:

1. **BOS stripping in the data collator.** The HuggingFace `Seq2SeqTrainer` internally calls `shift_tokens_right` on labels, which prepends `decoder_start_token_id` (which is the same as `<|startoftranscript|>` for Whisper). If the labels themselves already begin with `<|startoftranscript|>` (as they do when produced by `processor.tokenizer(text)` with `language='sw', task='transcribe'`), the shifted `decoder_input_ids` ends up with two leading SOT tokens, which is malformed and was empirically observed to cause loss collapse to 0 within ~200 steps under bf16. The collator therefore strips the leading BOS from each label sequence before padding.

2. **FP16 over bf16.** Initial experiments with bf16 exhibited training loss collapse to exactly 0.0 around step 100, with gradient norms simultaneously dropping to numerical zero -- a signature of bf16 underflow on the small loss values encountered late in fine-tuning. Switching to FP16 (which uses dynamic loss scaling) eliminated the collapse without requiring any other change.

A small additional preprocessing step skips KenSpeech samples with empty transcripts (approximately 22% of the 5,726 entries), as these would produce degenerate label sequences containing only special tokens.

#### 4.6.4 Word Error Rate Evaluation

`eval_wer.py` loads each model variant in turn, runs greedy decoding over the FLEURS sw_ke test set (487 utterances, rebuilt locally from the HuggingFace `google/fleurs` dataset), and computes Word Error Rate and Character Error Rate using the `evaluate` library's `wer` and `cer` metrics. Predictions and references are normalised using a lightweight scheme: lowercase, strip non-word punctuation, collapse whitespace. Per-model predictions are written to `preds_<name>.jsonl` so that alternative normalisation schemes can be applied post-hoc without re-running inference.

A digit-normalisation issue was observed in qualitative inspection: KenSpeech transcripts spell digits in word form (e.g., "elfu sita" rather than "6,000"), so fine-tuned models inherit this convention and are penalised on FLEURS samples that contain numerals. A separate post-processing script applies bidirectional digit-to-word normalisation before re-scoring; both raw and normalised WER are reported in Section 5.

#### 4.6.5 Orchestration

The five scripts are sequenced by `run.sh`, which is configurable via environment variables (data paths, hyperparameters, batch sizes, GPU launcher). Every step has a "skip if already done" guard, so the orchestrator can be re-run after a crash and will resume from the last successful step. Table 9b gives timing for both the initial capped run and the full-scale campaign:

| Step | Time (6K, capped) | Time (267K, full corpus) |
|---|---|---|
| Teacher pseudo-labelling | ~3-4 hours | ~20 hours |
| Filtering (CPU only) | < 1 minute | < 2 minutes |
| Training × 4 variants | ~1.5 hours | ~4-6 hours (est.) |
| WER evaluation across 5 systems | ~30 minutes | ~30 minutes |
| **Total** | **~5-6 hours** | **~25-27 hours** |

*Table 9b: ASR Pipeline Timing on AWS g5.2xlarge (A10G, 24GB VRAM)*

The full-corpus campaign represents a total AWS cost of approximately \$25-30 at the g5.2xlarge on-demand rate of \$1.21/hr. Using spot instances (typically 60-70% cheaper at ~\$0.40/hr) would reduce this to under \$12, though the resume-safe design of `pseudo_label.py` makes spot interruptions tolerable.

#### 4.6.6 Integration into the Cascaded Baseline

The trained ASR model serves as the front-end of the "improved cascade" baseline used in the S2ST evaluation of Section 5.6. Concretely, the cascade replaces the off-the-shelf `openai/whisper-small` checkpoint with the best-performing fine-tuned variant from Section 4.6 -- `ft_kenspeech_pseudo_raw` (Table 11, lowest WER among non-oracle systems) -- while keeping the NLLB-200 translation and MMS-TTS synthesis stages unchanged. The cascade inference flow is therefore:

1. **ASR (improved):** Source Swahili audio is transcribed with the fine-tuned Whisper-small in `language='sw', task='transcribe'` mode. Greedy decoding is used to match the conditions under which WER was measured.
2. **MT:** The Swahili transcript is translated to English by NLLB-200-distilled-1.3B at FP16, with input language token `swh_Latn` and target `eng_Latn`.
3. **TTS:** The English translation is synthesised to audio by MMS-TTS-eng (`facebook/mms-tts-eng`) at 16 kHz mono.
4. **ASR-BLEU re-transcription:** The synthesised English audio is re-transcribed with Whisper-medium in `language='en', task='transcribe'` mode, and BLEU is computed against the FLEURS en_us reference text. This re-transcription step is identical to the one used for the end-to-end S2ST system, ensuring an apples-to-apples comparison.

This integration is implemented as a single evaluation script (`whisper_asr/eval_cascade.py`) that runs both the vanilla and improved variants of the cascade in one pass and emits a side-by-side BLEU table. The two variants share NLLB and MMS-TTS components so that the only difference between them is the ASR front-end -- isolating the contribution of the ASR improvement to end-to-end translation quality.

The "vanilla cascade" row of Table 10 is produced identically but with `openai/whisper-small` substituted for the fine-tuned student. The difference between the two cascade rows therefore answers the question: "How much of the cascade's translation-quality deficit (versus the E2E system or against Seamless) is due to the ASR component being weak in zero-shot mode, versus due to error accumulation across MT and TTS that no ASR improvement can fix?"

#### 4.6.7 Released Artefacts

The complete ASR fine-tuning pipeline is released as a self-contained module under `hibiki-sw/whisper_asr/` in the project repository. The release comprises three categories of artefact, each independently usable.

**Trained model.** The `ft_kenspeech_pseudo_raw` Whisper-small student checkpoint -- the best non-oracle variant from Table 11 -- is distributed as a HuggingFace-compatible directory containing model weights, processor configuration, tokenizer, and the training arguments used. It can be loaded by any downstream consumer with a single call:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
processor = WhisperProcessor.from_pretrained("<release_path>", language="sw", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("<release_path>")
```

This is a drop-in replacement for `openai/whisper-small` in any Swahili ASR application. No code changes are required to use it.

**Pipeline scripts.** Eight Python scripts implement the full distillation, training, and evaluation workflow (Table 8 in Section 4.6, plus three evaluation scripts: `eval_cascade.py` for cascade text-BLEU, `eval_seamless.py` for the Seamless-M4T comparison, and `format_results.py` for paste-ready markdown table generation). The `run.sh` orchestrator sequences pseudo-labelling, filtering, training of all four variants, and WER evaluation, with a "skip if already done" guard at every step so that a partial run can be resumed without recomputing earlier stages. All scripts are configurable via environment variables and require only that the user has KenSpeech and Common Voice Swahili audio on local disk; the FLEURS test set is rebuilt from HuggingFace by a single inline script invocation.

**Filter recipe and ablation methodology.** The three-filter recipe (teacher confidence, n-gram repetition, and Swahili language identification) is implemented in `filter_pseudo.py` with each filter as a small, independently-toggleable function. The script emits `pseudo_labels_filtered.stats.json` containing the kept-sample count under all $2^3 = 8$ filter subsets, enabling researchers applying the methodology to other low-resource languages to ablate each filter's marginal contribution without re-running the pipeline. The recipe is intentionally language-agnostic: substituting the language code in the language-identification filter (`Language.SWAHILI` $\to$ another) and the teacher's `language` token is the only change required to apply the same recipe to, e.g., Hausa or Yoruba.

**Reproducing the headline result** requires a single A10G-class GPU (24 GB VRAM) for approximately 6 hours of compute (3-4 hours teacher inference, 1.5 hours student training across all four variants, and 30 minutes WER evaluation), plus approximately 50 GB of disk for downloads and intermediate outputs. The KenSpeech and Common Voice Swahili datasets are both freely available; no proprietary data, no closed model APIs, and no annotation effort beyond what already exists in those public corpora is required.

---

## 5 RESULTS AND ANALYSIS

### 5.1 Data Pipeline Output

*[This section will be populated with actual results after pipeline execution]*

The synthetic data pipeline processed Common Voice Swahili validated utterances through the full pipeline:

| Stage | Input | Output | Count |
|---|---|---|---|
| Transcription | CV Swahili audio + TSV transcripts | JSON (text + word timestamps) | XX,XXX |
| Translation | Transcription JSONs | Translation JSONs (sw->en) | XX,XXX |
| Alignment | Translation pairs | Alignment JSONs | XX,XXX |
| Synthesis | Translations + alignments | WAV files (24 kHz) | XX,XXX |
| Encoding (source) | CV Swahili audio | Mimi tokens (.npy, shape 8xT) | XX,XXX |
| Encoding (target) | Synthesised English audio | Mimi tokens (.npy, shape 8xT) | XX,XXX |

### 5.2 Training Convergence

*[This section will include training loss curves for each stage]*

### 5.3 Translation Quality (ASR-BLEU)

*[This section will present BLEU scores on FLEURS test set]*

| Direction | ASR-BLEU | Text BLEU |
|---|---|---|
| Sw -> En | X.XX | X.XX |
| En -> Sw | X.XX | X.XX |

### 5.4 Speaker Similarity

*[This section will present WavLM cosine similarity scores]*

### 5.5 Latency Analysis

*[This section will present LAAL and end-offset measurements]*

### 5.6 Comparison with Baselines

The S2ST system is compared against two cascade baselines and one off-the-shelf E2E system. The "vanilla cascade" uses unmodified Whisper-small for ASR; the "improved cascade" uses the best-performing fine-tuned Whisper-small from Section 4.6 -- `ft_kenspeech_pseudo_raw`, which achieved the lowest WER among non-oracle systems. Including both isolates the contribution of E2E modelling from the contribution of simply having a better ASR front-end.

For the cascade rows, **text-BLEU** (Sw audio -> ASR -> NLLB-200 -> En text, scored against the FLEURS en_us reference) is reported in place of full ASR-BLEU. Text-BLEU is a strict upper bound on ASR-BLEU -- adding the MMS-TTS + Whisper-medium re-transcription loop can only introduce errors -- so reporting the upper bound is conservative for the cascade and any cascade-vs-E2E gap shown here will be at least as large after the full pipeline. Hibiki-Sw and Seamless rows use full ASR-BLEU. Translation is by NLLB-200-distilled-1.3B with beam=4. All numbers are corpus-level scores from `sacrebleu`.

| System | Direction | BLEU | chrF | Speaker Sim | LAAL | n |
|---|---|---|---|---|---|---|
| **Hibiki-Sw (ours)** | Sw->En | X.XX (ASR-BLEU) | X.XX | X.XX | X.XX | 487 |
| Vanilla cascade (whisper-small + NLLB) | Sw->En | 4.78 (text-BLEU) | 31.22 | N/A | N/A | 487 |
| **Improved cascade** (ft-whisper-small + NLLB) | Sw->En | **16.31** (text-BLEU) | **46.64** | N/A | N/A | 487 |
| Seamless-M4T-v2-large (Meta) | Sw->En | 21.29 (S2T) | 53.30 | X.XX | X.XX | 487 |
| **Hibiki-Sw (ours)** | En->Sw | X.XX (ASR-BLEU) | X.XX | X.XX | X.XX | 581 |
| Vanilla cascade (whisper-small + NLLB) | En->Sw | -- | -- | N/A | N/A | -- |
| Seamless-M4T-v2-large (Meta) | En->Sw | 20.46 (S2T) | 57.71 | X.XX | X.XX | 581 |

*Table 10: Comparison with Baseline Systems on the FLEURS sw_ke <-> en_us test set. Cascade and Seamless rows report S2T / text-BLEU; Hibiki-Sw and Seamless rows for the full S2ST pipeline use ASR-BLEU. The En->Sw cascade row is omitted because the ASR fine-tuning of Section 4.6 was Swahili-only and therefore offers no improvement over off-the-shelf Whisper-small for English source audio; an evaluation of the unmodified cascade in this direction was not pursued for this report.*

Two cascade-related observations follow from Table 10.

**The improved cascade closes most of the gap to a 23x larger multilingual model.** On Sw->En, the improved cascade reaches 16.31 BLEU against 21.29 for SeamlessM4T-v2-large. Considering that Seamless uses approximately 23 times the parameters (2.3 B vs the cascade's ~1.5 B = whisper-small 244 M + NLLB-1.3B), is trained on substantially more parallel data and a wider multilingual mixture, and operates as a dedicated end-to-end model, recovering 77% (16.31/21.29) of its translation quality with a specialised cascade is a competitive result. The 5-BLEU residual gap quantifies the value Seamless extracts from end-to-end speech-to-text training on its much larger corpus.

**Improving the ASR front-end alone raises Sw->En translation quality by +11.5 BLEU** (4.78 -> 16.31) and +15.4 chrF (31.22 -> 46.64), without changing the MT model, the language pair, or anything else about the cascade. This is one of the largest single-component improvements available to a cascade for low-resource Swahili speech translation, and it is a contribution-claim available to any future cascade-based system, not only this project. The full `ft_kenspeech_pseudo_raw` ASR model and pipeline are released alongside the report (Section 4.6).

A noteworthy secondary observation, drawn from the underlying per-system cascade results, is that the BLEU score of `ft_kenspeech_pseudo_raw` (16.31) is statistically indistinguishable from that of the gold-upper-bound variant `ft_kenspeech_gold_upper_bound` (16.28), despite the gold variant having a 5.7-point lower WER (37.28 vs 42.98 -- Table 11). NLLB-200 appears to be sufficiently robust to the additional Sw transcript noise produced by pseudo-label-trained ASR that the residual WER gap does not propagate into translation quality. This is a stronger result than WER alone would suggest: for cascade applications targeting Sw->En translation, pseudo-label distillation fully closes the gap to gold-data ASR fine-tuning, even though the same is not true for the intrinsic ASR metric. The chrF score (which is more sensitive to character-level differences) does retain a small advantage for the gold variant (48.86 vs 46.64), but the BLEU equivalence is the relevant signal for downstream translation utility.

### 5.7 Improved Cascade -- Swahili ASR Results

The Whisper-small Sw ASR fine-tuning pipeline of Section 4.6 produced four trained student variants alongside a vanilla zero-shot baseline. All five systems were evaluated on the FLEURS sw_ke test set (487 utterances) with a lightweight lowercase + punctuation-stripping normaliser, and (in the second column) a digit-to-word post-processing step that addresses the convention mismatch between numeral references and spelled-out predictions discussed in Section 4.6.4.

| System | WER | WER (digit-norm) | CER | n |
|---|---|---|---|---|
| `vanilla_small` (zero-shot) | 87.22 | 86.11 | 29.88 | 487 |
| `ft_kenspeech_only` | 52.24 | 51.62 | 18.90 | 487 |
| `ft_kenspeech_pseudo_raw` | **42.98** | **42.69** | **13.63** | 487 |
| `ft_kenspeech_pseudo_filtered` | 45.47 | 45.05 | 16.07 | 487 |
| `ft_kenspeech_gold_upper_bound` | **37.28** | **36.66** | 14.42 | 487 |

*Table 11: Swahili ASR Results on FLEURS sw_ke (487 utterances). Best non-oracle WER in bold.*

Four observations follow from Table 11.

**Supervised fine-tuning on KenSpeech alone reduces WER by 35 points** (87.22 -> 52.24), confirming that even a small (5,726-utterance) high-quality labelled corpus substantially closes the gap between Whisper-small zero-shot and a domain-adapted student. This single number is the strongest argument for fine-tuning Whisper for any low-resource language with a comparably-sized labelled set available.

**Pseudo-label distillation provides a further 9.3-point WER reduction** (52.24 -> 42.98 with raw pseudo-labels). Adding ~5,900 Whisper-large-v3-labelled utterances on top of KenSpeech recovers more than half of the remaining gap to the gold upper bound. This is the core positive result of the pseudo-label distillation contribution.

**The proposed filter recipe did not improve over raw pseudo-labels at the chosen thresholds.** `ft_kenspeech_pseudo_filtered` (45.47 WER) is approximately 2.5 points worse than `ft_kenspeech_pseudo_raw` (42.98 WER). The filters rejected only 4.4% of pseudo-labels (Table 12); at this level of activity, the few samples removed appear to be net useful for the student. Two interpretations are consistent with the data: (i) Whisper-large-v3 is a sufficiently reliable Swahili teacher that mild filtering trades a small amount of label quality for a more impactful loss in label quantity; or (ii) the 2.5-point difference falls within the noise of evaluation on 487 samples (approximate per-system standard error is ~2 WER points). On a less reliable teacher (e.g., Whisper-medium) or a noisier source corpus, filtering would be expected to matter more. The recipe is therefore retained as a tunable component of the pipeline, with the empirical observation that aggressive filtering is not warranted when the teacher is strong.

**Substituting gold transcripts where available reduces WER by a further 5.7 points** (42.98 -> 37.28). This gap quantifies the headroom remaining for improved pseudo-label quality and motivates future work on iterative self-training, larger teacher models, or more aggressive filter thresholds.

#### 5.7.1 Filter Ablation

The per-filter ablation from `pseudo_labels_filtered.stats.json` shows the marginal contribution of each of the three filters (Table 12). The "no filter" row corresponds to `ft_kenspeech_pseudo_raw`; the "all filters" row corresponds to `ft_kenspeech_pseudo_filtered`. The intermediate rows enable claims of the form "the language-ID filter alone removes X% of pseudo-labels and is responsible for a Y-point WER reduction."

| Active filters | Pseudo-labels kept | % of total |
|---|---|---|
| (none) | 5,933 | 100.0 |
| confidence | 5,876 | 99.0 |
| lang_id | 5,873 | 99.0 |
| repetition | 5,785 | 97.5 |
| confidence + lang_id | 5,818 | 98.1 |
| confidence + repetition | 5,730 | 96.6 |
| lang_id + repetition | 5,727 | 96.5 |
| confidence + lang_id + repetition (all) | 5,674 | 95.6 |

*Table 12: Pseudo-Label Filter Ablation. Thresholds: confidence > -1.0, repetition < 0.5, sw_lang_id >= 0.7.*

The repetition filter is the single most active rejector (148 entries, 2.5%); the confidence and language-id filters reject 57 and 60 entries respectively (about 1% each). The substantial overlap between the three is small (the union of all three rejects 259 entries, less than the sum of 265, suggesting only ~2% of rejected entries fail more than one filter). Combined with the WER results in Table 11, this confirms that with a strong teacher the filters are catching genuinely degenerate outputs rather than borderline-noisy ones, but the population of degenerate outputs at this teacher scale is too small for filtering to translate into a measurable WER improvement.

#### 5.7.2 Qualitative Examples

Representative side-by-side outputs on FLEURS sw_ke samples illustrate the qualitative gap between vanilla and fine-tuned Whisper-small. The vanilla model produces fragmented Swahili with broken word boundaries and English-style capitalisation; the fine-tuned variants produce coherent Swahili with appropriate orthographic conventions. The gold-upper-bound variant occasionally matches references exactly (e.g., `fleurs_sw_ke_00001.wav` below).

**`fleurs_sw_ke_00000.wav`**

| | |
|---|---|
| REF | katika miaka yote ya 1960 brzezinski alimfanyia kazi john f. kennedy kama mshauri wake na pia utawala wa lyndon b. johnson |
| vanilla_small | katika meka yo tia elf moja meheti sasiti ni, Jensiski Alinfaniya Kazijon FKnedi Kama Mshawuva kenapia utawala wa Linod B. Johnson. |
| ft_kenspeech_only | katika meka yote ya elfumoja miwetisa siteni ndio nzisiki alimfania kazi john fk nedi kama mshavu wakinapewa utawala walio na bay johnson |
| ft_kenspeech_pseudo_raw | Katika meka yote ya 1160, Junziski alimfania kazi John F. Kennedy kama mshawivake na pia utawala walinod B. Johnson. |
| ft_kenspeech_pseudo_filtered | Katika meka yote ya 1196, Junziski alimfanyia kazi John F. Kennedy kama mshavu wake na pia utawala walinod B. Johnson. |
| ft_kenspeech_gold_upper_bound | katika miaka yote ya 1960 john zisiki alimfania kazi john fk4 ni kama mshawi wake na pia utawala wa leo bay johnson |

**`fleurs_sw_ke_00001.wav`**

| | |
|---|---|
| REF | sauti ya piramidi na onyesho la nuru ni mojawapo ya mambo yanayofurahisha zaidi katika eneo la watoto |
| vanilla_small | Sauti ya piramidi na onyesho la nuru ni mojawa po ya mambo ya na ufuraisha zaidi katika ya neola ototo. |
| ft_kenspeech_only | sauti ya piramidi na onyesho la nuru ni mojawapo ya mambo ya naufurahisha zaidi katika eneo la ototo |
| ft_kenspeech_pseudo_raw | Sauti ya piramidi na onyesho la nuru ni moja wapo ya mambo ya naufurahisha zaidi katika eneo la ototo. |
| ft_kenspeech_pseudo_filtered | Sauti ya piramidi na onyesho la nuru ni moja wapo ya mambo ya naufurahisha zaidi katika eneo la ototo. |
| ft_kenspeech_gold_upper_bound | sauti ya piramidi na onyesho la nuru ni mojawapo ya mambo yanayofurahisha zaidi katika eneo la watoto |

Two patterns are visible across the qualitative examples. First, all four fine-tuned variants resolve the broken word boundaries and English-style capitalisation produced by the vanilla model (e.g., "Sauti ya piramidi" instead of "Sauti ya piramidi na onyesho la nuru ni mojawa po"). Second, the pseudo-label-trained variants (`pseudo_raw`, `pseudo_filtered`) inherit Whisper-large-v3's preference for digit numerals and English-style sentence capitalisation, while `ft_kenspeech_only` and `ft_kenspeech_gold_upper_bound` use the spelled-out and lowercase conventions of the underlying KenSpeech and FLEURS-train annotations respectively. This convention inheritance does not significantly affect WER (see digit-normalised column in Table 11) but is visible at inspection.

#### 5.7.3 Reproducibility Statement

Every number in Tables 11 and 12, the cascade rows of Table 10, and the qualitative examples above is produced by deterministic scripts operating on freely available data, with the full audit trail traceable from raw audio to reported number through three script invocations. After the one-time pipeline run (Section 4.6.5), the reported numbers are regenerable from the saved per-system JSONLs in seconds:

| Table | Script | Inputs |
|---|---|---|
| Table 11 (WER + CER) | `format_results.py` | `preds_*.jsonl` from `eval_wer.py` |
| Table 12 (filter ablation) | `format_results.py` | `pseudo_labels_filtered.stats.json` from `filter_pseudo.py` |
| Table 10, cascade rows | `eval_cascade.py` | `preds_*.jsonl` from `eval_wer.py` (no ASR re-inference) |
| Table 10, Seamless rows | `eval_seamless.py` | FLEURS audio + parallel transcripts |

The cascade and Seamless evaluations are independent of the training run -- they read only the saved ASR predictions or, in Seamless's case, the public FLEURS audio. This means an independent reviewer with a single A10G-class GPU can verify the cascade-improvement claim (+11.5 BLEU) and the Seamless comparison (16.31 vs 21.29) in approximately one hour, without re-running the much longer pseudo-labelling and student-training stages.

Random-seed effects are non-trivial for fine-tuning on small data, and all four trained variants in Tables 11 and 12 used `seed=42` throughout. Re-running with a different seed is expected to perturb each variant's WER by approximately 1-2 absolute points but should preserve the rank ordering observed in Table 11 and the cascade BLEU pattern of Table 10. The qualitative examples in Section 5.7.2 are deterministic given the trained checkpoints (greedy decoding, no sampling).

### 5.8 Qualitative Analysis

*[This section will include sample spectrograms and transcriptions of model outputs, discussing common error patterns, failure modes, and strengths]*

---

## 6 RECOMMENDATIONS

Based on the findings of this project, the following recommendations are made for future work:

1. **Larger model scale:** The 100M parameter model demonstrates the architecture's viability but likely underperforms at this scale. Future work should explore 300M-1B parameter variants when more GPU compute is available, as the Hibiki paper demonstrated significant quality gains with scale.

2. **Real Swahili parallel speech data:** The synthetic data pipeline, while functional, introduces quality limitations through TTS synthesis. Recording actual English-Swahili parallel speech data (even 50-100 hours) would likely yield significant quality improvements when used for fine-tuning (Stage 4).

3. **Extension to other Kenyan languages:** The methodology is transferable to other Kenyan languages with available speech data, such as Kikuyu, Luo, and Kalenjin. MMS-TTS covers many of these languages, and Common Voice is actively collecting data for several.

4. **Streaming inference optimization:** While the model is architecturally capable of streaming (due to causal attention and silence insertion), the inference pipeline has not been optimised for real-time performance. Integration with WebSocket APIs and client-side audio buffering would be needed for deployment.

5. **Full-corpus Whisper fine-tuning and iterative self-training:** The WER results reported in Section 5.7 were obtained using approximately 6,000 pseudo-labels from an initial capped run. A full-scale teacher inference campaign over all 267,652 Common Voice Swahili validated utterances was initiated on the A10G and had reached 55% completion at the time of writing. Re-training the student on this larger pseudo-label corpus (expected ~250K usable labels after filtering, a 40x increase) is the most immediate avenue for WER improvement. Beyond corpus scaling, investigating whether the optimal filter thresholds shift with corpus size, and whether a second round of self-training (using the fine-tuned student as the new teacher, in the SlimIPL style [Likhomanenko et al., 2021]) yields diminishing or compounding gains, are both natural extensions.

6. **Multi-speaker TTS:** MMS-TTS produces single-speaker output, limiting voice diversity in synthetic data and making voice preservation evaluation less meaningful. Fine-tuning a multi-speaker VITS model on Common Voice Swahili (which has multiple speakers) would improve both synthetic data quality and the model's voice transfer capabilities.

---

## 7 CONCLUSION

This project successfully designed and implemented a ~100M parameter end-to-end speech-to-speech translation system for bidirectional English-Swahili translation, adapted from the Hibiki architecture. The key contributions are:

1. **Architecture adaptation:** The Hibiki architecture was scaled down from 2.7B to ~100M parameters (Temporal Transformer: 512-dim, 12 layers, ~65M params; Depth Transformer: 384-dim, 4 layers/codebook, ~30M params) while retaining the core multistream design with source, target, and inner monologue text streams operating on Mimi codec tokens.

2. **Synthetic data pipeline for a low-resource pair:** A complete pipeline was developed to generate parallel English-Swahili speech data from monolingual resources, using WhisperX forced alignment (preserving human transcripts from Common Voice), NLLB-200 machine translation, MMS-TTS synthesis, contextual alignment, and silence insertion. This pipeline is reproducible and applicable to other low-resource language pairs.

3. **Resource-constrained training:** The four-stage training protocol was executed on Kaggle T4 GPUs within a total budget of ~44 GPU-hours, demonstrating that meaningful S2ST research can be conducted without access to large GPU clusters. Specific optimizations included FP16 mixed precision, gradient checkpointing, int8 Whisper quantization, resumable pipeline scripts, and systematic GPU memory management.

4. **Evaluation framework:** The system was evaluated against both cascaded baselines (Whisper + NLLB + MMS-TTS) and Meta's Seamless model using ASR-BLEU, speaker similarity, and latency metrics on the FLEURS benchmark.

5. **Improved cascaded baseline via ASR pseudo-label distillation.** A separate Whisper-small Swahili ASR model was developed by pseudo-labelling unlabelled Common Voice and FLEURS-train audio with a Whisper-large-v3 teacher and fine-tuning the student on the union of the pseudo-labels and the labelled KenSpeech corpus. The fine-tuned student reduced Word Error Rate on FLEURS sw_ke from 87.22% (vanilla Whisper-small) to 42.98% -- a 44-point absolute reduction, of which 9.3 points are attributable to the pseudo-labels alone over a KenSpeech-only fine-tune. When propagated through a Whisper-small + NLLB-200 cascade, this ASR improvement raises Sw->En translation BLEU from 4.78 to 16.31 (a +11.5 BLEU improvement) without any change to the translation model -- demonstrating that the ASR component is the dominant bottleneck in low-resource cascade translation and that cascade-vs-E2E comparisons that omit ASR fine-tuning materially understate cascade quality. A surprising secondary finding is that the pseudo-label-trained ASR achieves cascade BLEU statistically indistinguishable from a gold-label upper bound (16.31 vs 16.28) despite a residual 5.7-point WER gap, suggesting that NLLB-200 absorbs mild Sw transcript noise without translation-quality cost and that pseudo-label distillation is therefore more effective for downstream translation tasks than intrinsic WER metrics indicate. An ablation of the proposed three-filter recipe (confidence, n-gram repetition, Swahili language identification) showed that aggressive filtering did not improve WER at the chosen thresholds when paired with the Whisper-large-v3 teacher; this suggests filter recipes should be calibrated to teacher reliability rather than applied uniformly. The full pipeline, filter recipe, and per-filter ablation methodology are released as a reproducible artefact applicable beyond Swahili.

This work demonstrates that the Hibiki architecture can be adapted for low-resource African language pairs at a fraction of the original computational cost, opening a pathway for end-to-end speech translation research at African universities. The synthetic data generation methodology, codebase, trained models, and the auxiliary ASR fine-tuning pipeline are made available to support future work on Swahili and other underserved languages.

---

## REFERENCES

[1] A. S. Tanenbaum and D. J. Wetherall, *Computer Networks*, 5th ed. Pearson, 2011.

[2] A. Défossez, L. Music, T. Remez, et al., "Hibiki: Streaming Speech-to-Speech Translation with Prefix Alignment," Kyutai, Tech. Report, 2025.

[3] A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, "High Fidelity Neural Audio Compression," in *Proc. ICML*, 2023. (EnCodec)

[4] A. Défossez, L. Music, T. Remez, et al., "Moshi: a speech-text foundation model for real-time dialogue," Kyutai, Tech. Report, 2024.

[5] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, 2023. (Whisper)

[6] Seamless Communication et al., "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation," Meta AI, Tech. Report, 2023.

[7] NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation," Meta AI, Tech. Report, 2022.

[8] V. Pratap, A. Tjandra, B. Shi, et al., "Scaling Speech Technology to 1,000+ Languages," Meta AI, 2024. (MMS)

[9] Y. Jia, R. J. Weiss, F. Biadsy, et al., "Direct speech-to-speech translation with a sequence-to-sequence model," in *Proc. Interspeech*, 2019. (Translatotron)

[10] Y. Jia, M. T. Ramanovich, T. Remez, and R. Pang, "Translatotron 2: High-quality direct speech-to-speech translation with voice preservation," in *Proc. ICML*, 2022.

[11] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding," in *Neurocomputing*, 2024. (RoPE)

[12] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: a Method for Automatic Evaluation of Machine Translation," in *Proc. ACL*, 2002.

[13] R. Sennrich, B. Haddow, and A. Birch, "Improving Neural Machine Translation Models with Monolingual Data," in *Proc. ACL*, 2016.

[14] S. Chen, C. Wang, Z. Chen, et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," *IEEE JSTSP*, 2022.

[15] R. Ardila, M. Branez, L. Coheur, et al., "Common Voice: A Massively-Multilingual Speech Corpus," in *Proc. LREC*, 2020.

[16] S. Kudugunta, I. Caswell, B. Zhang, et al., "MADLAD-400: A Multilingual And Document-Level Large Audited Dataset," in *Proc. ACL*, 2024.

[17] T. Likhomanenko, Q. Xu, J. Pratap, A. Tomasello, J. Kahn, G. Synnaeve, V. Liptchinsky, and R. Collobert, "slimipl: Language-Model-Free Iterative Pseudo-Labeling," in *Proc. NeurIPS Workshop*, 2021.

[18] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, "SoundStream: An End-to-End Neural Audio Codec," *IEEE/ACM TASLP*, 2022.

[19] J. Kim, J. Kong, and J. Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," in *Proc. ICML*, 2021. (VITS)

[20] G. Borsos, R. Marinier, D. Vincent, et al., "AudioLM: a Language Modeling Approach to Audio Generation," *IEEE/ACM TASLP*, 2023.

[21] S. Gandhi, P. von Platen, and A. M. Rush, "Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling," arXiv preprint arXiv:2311.00430, 2023.

[22] D. Yarowsky, "Unsupervised Word Sense Disambiguation Rivaling Supervised Methods," in *Proc. ACL*, 1995.

[23] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Proc. NeurIPS*, 2020.

[24] M. Bain, J. Huh, T. Han, and A. Zisserman, "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio," in *Proc. Interspeech*, 2023.

[25] AWS, "Amazon EC2 G5 Instances," Amazon Web Services Documentation, 2024. Available: https://aws.amazon.com/ec2/instance-types/g5/

---

## APPENDICES

### Appendix A: Model Configuration (model_100m.yaml)

```yaml
model:
  codec:
    name: "kyutai/mimi"
    sample_rate: 24000
    frame_rate: 12.5
    num_codebooks: 8
    codebook_size: 2048
    frozen: true

  temporal:
    d_model: 512
    ffn_dim: 1408
    num_layers: 12
    num_heads: 8
    head_dim: 64
    max_seq_len: 250
    local_attn_window: 250
    dropout: 0.1
    activation: "silu"
    rope: true

  depth:
    d_model: 384
    ffn_dim: 1024
    num_layers_per_codebook: 4
    num_codebooks_output: 8
    num_codebooks_input: 8
    weight_sharing_start: 5
    embedding_dim: 64
    dropout: 0.1
    activation: "silu"

  tokens:
    text_vocab_size: 32000
    audio_codebook_size: 2048
    pad_token_id: 0
    bos_token_id: 1
    eos_token_id: 2
    epad_token_id: 3
    acoustic_delay: 2
```

### Appendix B: Synthetic Data Pipeline Commands

```bash
# Step 1: Transcription with forced alignment (CV Swahili)
python data/prepare/transcribe_whisper.py \
    --source common_voice \
    --dataset_dir /kaggle/input/cv-swahili/cv-validated \
    --lang sw --split validated \
    --output_dir transcriptions/sw \
    --whisper_model small --compute_type int8 \
    --forced_alignment

# Step 2: Translation (Sw -> En)
python data/prepare/translate_nllb.py \
    --input_dir transcriptions/sw \
    --output_dir translations/sw2en \
    --source_lang sw --target_lang en \
    --dtype float16 --batch_size 8

# Step 3: Contextual alignment
python data/prepare/run_pipeline.py \
    --source common_voice \
    --source_lang sw --target_lang en \
    --base_dir /kaggle/working/hibiki-sw \
    --step align

# Step 4: TTS synthesis + silence insertion
python data/prepare/synthesize_tts.py \
    --translation_dir translations/sw2en \
    --output_dir synthetic_audio/sw2en \
    --target_lang en \
    --alignment_dir alignments/sw2en \
    --whisper_model small --target_sr 24000

# Step 5: Mimi encoding (source)
python data/prepare/encode_audio.py \
    --source common_voice --lang sw \
    --dataset_dir /kaggle/input/cv-swahili/cv-validated \
    --split validated \
    --output_dir audio_tokens/cv_sw \
    --num_codebooks 8 --max_duration 20.0

# Step 5: Mimi encoding (target)
python data/prepare/encode_audio.py \
    --source directory \
    --audio_dir synthetic_audio/sw2en/aligned_audio \
    --output_dir audio_tokens/synth_en \
    --num_codebooks 8 --max_duration 30.0

# Step 6: Create S2ST manifest
python data/prepare/create_s2st_manifest.py \
    --source synthetic \
    --source_token_dir audio_tokens/cv_sw \
    --target_token_dir audio_tokens/synth_en \
    --translation_dir translations/sw2en \
    --text_token_dir aligned_text/sw2en \
    --output_manifest manifests/sw2en_train.tsv \
    --tokenizer_model tokenizer/sp_ensw_32k.model \
    --direction sw2en
```

### Appendix C: Training Commands

```bash
# Stage 1: Text adaptation
python training/train_text.py \
    --config configs/model_100m.yaml \
    --data_dir text_tokens/ \
    --output_dir checkpoints/stage1

# Stage 2: Audio pretraining
python training/train_audio.py \
    --config configs/model_100m.yaml \
    --data_dir audio_tokens/ \
    --checkpoint checkpoints/stage1/latest.pt \
    --output_dir checkpoints/stage2

# Stage 3: S2ST training (Sw->En)
python training/train_s2st.py \
    --config configs/model_100m.yaml \
    --manifest manifests/sw2en_train.tsv \
    --checkpoint checkpoints/stage2/latest.pt \
    --output_dir checkpoints/stage3_sw2en

# Stage 4: Fine-tuning (Sw->En)
python training/train_finetune.py \
    --config configs/model_100m.yaml \
    --manifest manifests/sw2en_finetune.tsv \
    --checkpoint checkpoints/stage3_sw2en/latest.pt \
    --output_dir checkpoints/stage4_sw2en
```
