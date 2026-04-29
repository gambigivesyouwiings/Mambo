#!/usr/bin/env bash
# Overnight orchestrator: 4 training runs + 7 inference variants + BLEU eval.
#
# Designed for AWS EC2 g6.12xlarge (4x L4 GPUs, 96 GB total VRAM, native bf16).
# Runs sequentially because each training job uses all 4 GPUs via DDP.
#
# Usage (from the hibiki-sw repo root, one-shot in the background):
#     mkdir -p /data/runs
#     nohup ./whisper_st/run_overnight.sh > /data/runs/orchestrator.log 2>&1 &
#     tail -f /data/runs/orchestrator.log
#
# Override paths via env vars before running:
#     DATA_ROOT=/mnt/data ./whisper_st/run_overnight.sh
#
# Recovery: every step has a "skip if already done" guard, so you can rerun
# the script after a partial failure and it'll resume from where it stopped.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------

DATA_ROOT=${DATA_ROOT:-/data}
RUN_ROOT=${RUN_ROOT:-$DATA_ROOT/runs}

# Inputs
KENSPEECH_DIR=${KENSPEECH_DIR:-$DATA_ROOT/kenspeech-sw}
SW2EN_TRANSLATIONS=${SW2EN_TRANSLATIONS:-$DATA_ROOT/sw2en/translations/sw2en}
SW2EN_ALIGNMENTS=${SW2EN_ALIGNMENTS:-$DATA_ROOT/sw2en/alignments/sw2en}
EN2SW_ALIGNMENTS=${EN2SW_ALIGNMENTS:-$DATA_ROOT/en2sw/alignments/en2sw}

# Outputs
LEXICON=${LEXICON:-$RUN_ROOT/lexicon.jsonl}
TEST_DIR=${TEST_DIR:-$RUN_ROOT/fleurs_sw_ke_test}
PRED_DIR=${PRED_DIR:-$RUN_ROOT/predictions}

# Training hyperparameters
EPOCHS=${EPOCHS:-3}
WARMUP=${WARMUP:-100}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16}    # set to fp16 for T4 / older GPUs

# DDP launcher (4-GPU L4 default; for Kaggle 2x T4 set:
#   LAUNCH="accelerate launch --num_processes 2 --mixed_precision fp16"
# or for single-GPU debug:
#   LAUNCH="python")
LAUNCH=${LAUNCH:-"accelerate launch --num_processes 4 --mixed_precision bf16"}

# Skip flags for resource-constrained environments (e.g. Kaggle 2x T4 in 12 hrs)
SKIP_MEDIUM=${SKIP_MEDIUM:-0}    # set to 1 to skip Whisper-medium variants

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

mkdir -p "$RUN_ROOT" "$PRED_DIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_ROOT/orchestrator.log"; }

assert_dir() {
    if [[ ! -d "$1" ]]; then
        log "ERROR: required directory missing: $1"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

log "=== Pre-flight ==="
log "DATA_ROOT=$DATA_ROOT"
log "RUN_ROOT=$RUN_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RUN_ROOT/orchestrator.log"

assert_dir "$KENSPEECH_DIR"
assert_dir "$SW2EN_TRANSLATIONS"
assert_dir "$SW2EN_ALIGNMENTS"
assert_dir "$EN2SW_ALIGNMENTS"

# -----------------------------------------------------------------------------
# Step 1: Build bidirectional lexicon
# -----------------------------------------------------------------------------

log "=== Step 1: Build lexicon ==="
if [[ -s "$LEXICON" ]]; then
    log "  Lexicon already exists at $LEXICON ($(wc -l < "$LEXICON") entries) — skipping"
else
    python whisper_st/build_lexicon.py \
        --alignments sw2en="$SW2EN_ALIGNMENTS" en2sw="$EN2SW_ALIGNMENTS" \
        --output_path "$LEXICON" \
        --min_freq 2 --min_alignment_consistency 0.4 \
        --include_wikidata 2>&1 | tee -a "$RUN_ROOT/lexicon_build.log"
    log "  Lexicon built: $(wc -l < "$LEXICON") entries"
fi

# -----------------------------------------------------------------------------
# Step 2: Build FLEURS sw_ke test reference set
# -----------------------------------------------------------------------------

log "=== Step 2: Build FLEURS sw_ke test set ==="
if [[ -s "$TEST_DIR/refs.jsonl" ]]; then
    log "  Test set already built ($(wc -l < "$TEST_DIR/refs.jsonl") refs) — skipping"
else
    mkdir -p "$TEST_DIR/audio"
    python - <<EOF 2>&1 | tee -a "$RUN_ROOT/fleurs_build.log"
import json, os
import soundfile as sf, numpy as np
from datasets import load_dataset

sw_ds = load_dataset('google/fleurs', 'sw_ke', split='test', trust_remote_code=True)
en_ds = load_dataset('google/fleurs', 'en_us', split='test', trust_remote_code=True)
en_by_id = {s['id']: s['transcription'] for s in en_ds}

n = 0
with open('$TEST_DIR/refs.jsonl', 'w', encoding='utf-8') as f:
    for i, sample in enumerate(sw_ds):
        sid = sample['id']
        if sid not in en_by_id:
            continue
        wav_name = f'fleurs_sw_ke_{i:05d}.wav'
        sf.write(f'$TEST_DIR/audio/{wav_name}',
                 np.asarray(sample['audio']['array'], dtype=np.float32),
                 sample['audio']['sampling_rate'])
        f.write(json.dumps({
            'audio': wav_name,
            'reference_en': en_by_id[sid],
            'reference_sw': sample['transcription'],
            'id': sid,
        }, ensure_ascii=False) + '\n')
        n += 1
print(f'Wrote {n} parallel test pairs')
EOF
    log "  Test set built: $(wc -l < "$TEST_DIR/refs.jsonl") refs"
fi

# -----------------------------------------------------------------------------
# Step 3: Four training runs
# -----------------------------------------------------------------------------

# Each variant: (name, base_model, batch_size, grad_accum, lr, lexicon_arg)
# - hint=on means train.py is given --lexicon_path so model learns the prompt format

train_variant() {
    local NAME=$1
    local BASE=$2
    local BS=$3
    local GA=$4
    local LR=$5
    local LEX_FLAG=$6
    local OUT="$RUN_ROOT/$NAME"

    log "--- Training: $NAME ($BASE, bs=$BS, ga=$GA, lr=$LR, hint=${LEX_FLAG:+on}${LEX_FLAG:-off}) ---"
    if [[ -f "$OUT/final/config.json" ]]; then
        log "    Already trained at $OUT/final — skipping"
        return 0
    fi

    $LAUNCH whisper_st/train.py \
        --base_model "$BASE" \
        --translations_dir "$SW2EN_TRANSLATIONS" \
        --kenspeech_dir "$KENSPEECH_DIR" \
        --output_dir "$OUT" \
        --batch_size "$BS" --grad_accum "$GA" \
        --lr "$LR" --epochs "$EPOCHS" --warmup_steps "$WARMUP" \
        --ctc_loss_weight 0.3 \
        --precision "$PRECISION" \
        --num_workers 4 \
        --seed "$SEED" \
        $LEX_FLAG 2>&1 | tee "$OUT.log"
    log "    Done: $NAME"
}

log "=== Step 3: Training (variants) ==="
train_variant "whisper_small_baseline"  "openai/whisper-small"   8 1 1e-5 ""
train_variant "whisper_small_hint"      "openai/whisper-small"   8 1 1e-5 "--lexicon_path $LEXICON --hint_prob 0.5"

if [[ "$SKIP_MEDIUM" -eq 1 ]]; then
    log "  SKIP_MEDIUM=1 -- skipping Whisper-medium variants"
else
    train_variant "whisper_medium_baseline" "openai/whisper-medium"  4 2 8e-6 ""
    train_variant "whisper_medium_hint"     "openai/whisper-medium"  4 2 8e-6 "--lexicon_path $LEXICON --hint_prob 0.5"
fi

# -----------------------------------------------------------------------------
# Step 4: Inference (1 vanilla baseline + 4 trained models x 2 inference modes = 9)
# -----------------------------------------------------------------------------

run_inference() {
    local NAME=$1
    local CMD=$2
    local OUT="$PRED_DIR/preds_${NAME}.jsonl"
    log "--- Inference: $NAME ---"
    if [[ -s "$OUT" ]]; then
        log "    Predictions already exist at $OUT — skipping"
        return 0
    fi
    eval "$CMD --output_path $OUT" 2>&1 | tee "$PRED_DIR/${NAME}.log"
    log "    Done: $NAME ($(wc -l < "$OUT") predictions)"
}

log "=== Step 4: Inference ==="

# Vanilla zero-shot baseline (single GPU is fine)
run_inference "vanilla_small" \
    "python whisper_st/baseline_vanilla.py \
        --base_model openai/whisper-small \
        --audio_dir $TEST_DIR/audio"

VARIANTS="whisper_small_baseline whisper_small_hint"
if [[ "$SKIP_MEDIUM" -ne 1 ]]; then
    VARIANTS="$VARIANTS whisper_medium_baseline whisper_medium_hint"
fi

# Trained variants × {no-lex, lex} inference modes
for VARIANT in $VARIANTS; do
    MODEL_DIR="$RUN_ROOT/$VARIANT/final"
    if [[ ! -d "$MODEL_DIR" ]]; then
        log "  Skipping $VARIANT (model dir missing)"
        continue
    fi

    # No-lexicon inference
    run_inference "${VARIANT}_nolex" \
        "python whisper_st/inference.py \
            --model_dir $MODEL_DIR \
            --audio_dir $TEST_DIR/audio"

    # Lexicon-augmented inference
    run_inference "${VARIANT}_lex" \
        "python whisper_st/inference.py \
            --model_dir $MODEL_DIR \
            --audio_dir $TEST_DIR/audio \
            --lexicon_path $LEXICON"
done

# -----------------------------------------------------------------------------
# Step 5: BLEU eval over everything
# -----------------------------------------------------------------------------

log "=== Step 5: BLEU eval ==="

# Build the predictions arg list dynamically
PRED_ARGS=""
for f in "$PRED_DIR"/preds_*.jsonl; do
    [[ -s "$f" ]] || continue
    NAME=$(basename "$f" .jsonl | sed 's/^preds_//')
    PRED_ARGS="$PRED_ARGS ${NAME}=${f}"
done

if [[ -z "$PRED_ARGS" ]]; then
    log "  ERROR: no prediction files found in $PRED_DIR"
    exit 1
fi

python whisper_st/eval_bleu.py \
    --references_path "$TEST_DIR/refs.jsonl" \
    --predictions $PRED_ARGS \
    --show_examples 12 \
    2>&1 | tee "$RUN_ROOT/results.txt"

log "=== ALL DONE at $(date) ==="
log "Results: $RUN_ROOT/results.txt"
log "Disk usage:"
du -sh "$RUN_ROOT" | tee -a "$RUN_ROOT/orchestrator.log"
