#!/usr/bin/env bash
# Overnight orchestrator for Sw ASR experiments.
#
# Pipeline:
#   1. Pseudo-label CV-Sw + FLEURS-sw-train with whisper-large-v3 (teacher)
#   2. Filter pseudo-labels (confidence + repetition + lang-id)
#   3. Build a "gold-when-available" variant of the filtered set (upper bound)
#   4. Train 4 Whisper-small variants
#   5. WER eval on FLEURS sw_ke (pre-built test set from earlier run)
#
# Resume: every step has a "skip if done" guard. Re-run after a crash to continue.
#
# Usage:
#     mkdir -p /home/ec2-user/data/asr_runs
#     nohup ./whisper_asr/run.sh > /home/ec2-user/data/asr_runs/orchestrator.log 2>&1 &
#     tail -f /home/ec2-user/data/asr_runs/orchestrator.log

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------

DATA_ROOT=${DATA_ROOT:-/home/ec2-user/data}
ASR_ROOT=${ASR_ROOT:-$DATA_ROOT/asr_runs}

# Inputs
KENSPEECH_DIR=${KENSPEECH_DIR:-$DATA_ROOT/kenspeech/kenspeech-sw}
FLEURS_TEST_DIR=${FLEURS_TEST_DIR:-$DATA_ROOT/runs/fleurs_sw_ke_test}

# Outputs
PSEUDO_DIR=${PSEUDO_DIR:-$ASR_ROOT/pseudo}
PSEUDO_RAW=${PSEUDO_RAW:-$PSEUDO_DIR/pseudo_labels.jsonl}
PSEUDO_FILTERED=${PSEUDO_FILTERED:-$PSEUDO_DIR/pseudo_labels_filtered.jsonl}
PSEUDO_GOLD=${PSEUDO_GOLD:-$PSEUDO_DIR/pseudo_labels_gold.jsonl}

# Pseudo-labeling caps (use to keep teacher inference under budget)
MAX_PER_SOURCE=${MAX_PER_SOURCE:-}   # empty = unlimited
CV_VERSION=${CV_VERSION:-17_0}
TEACHER_MODEL=${TEACHER_MODEL:-openai/whisper-large-v3}

# Filter thresholds
CONF_THRESH=${CONF_THRESH:--1.0}
REP_THRESH=${REP_THRESH:-0.5}
SW_LID_THRESH=${SW_LID_THRESH:-0.7}

# Training hyperparameters
STUDENT_MODEL=${STUDENT_MODEL:-openai/whisper-small}
EPOCHS=${EPOCHS:-3}
WARMUP=${WARMUP:-200}
LR=${LR:-1e-5}
BS=${BS:-16}
GA=${GA:-1}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16}

# DDP launcher (4-GPU L4 default; for single-GPU debug: LAUNCH="python")
LAUNCH=${LAUNCH:-"accelerate launch --num_processes 4 --mixed_precision bf16"}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

mkdir -p "$ASR_ROOT" "$PSEUDO_DIR"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ASR_ROOT/orchestrator.log"; }

assert_dir() {
    if [[ ! -d "$1" ]]; then
        log "ERROR: required directory missing: $1"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------

log "=== Pre-flight ==="
log "DATA_ROOT=$DATA_ROOT  ASR_ROOT=$ASR_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    | tee -a "$ASR_ROOT/orchestrator.log" \
    || log "  (nvidia-smi not in PATH — GPU check skipped)"

assert_dir "$KENSPEECH_DIR"
assert_dir "$FLEURS_TEST_DIR/audio"
[[ -s "$FLEURS_TEST_DIR/refs.jsonl" ]] || { log "ERROR: missing $FLEURS_TEST_DIR/refs.jsonl"; exit 1; }

# -----------------------------------------------------------------------------
# Step 1: Pseudo-label with teacher
# -----------------------------------------------------------------------------

log "=== Step 1: Pseudo-label with $TEACHER_MODEL ==="
if [[ -s "$PSEUDO_RAW" ]]; then
    log "  $PSEUDO_RAW exists ($(wc -l < "$PSEUDO_RAW") entries) — pseudo_label.py will resume from it"
fi
MAX_FLAG=""
[[ -n "$MAX_PER_SOURCE" ]] && MAX_FLAG="--max_per_source $MAX_PER_SOURCE"

python whisper_asr/pseudo_label.py \
    --out_dir "$PSEUDO_DIR" \
    --teacher_model "$TEACHER_MODEL" \
    --sources fleurs_sw_train cv_sw \
    --cv_version "$CV_VERSION" \
    --precision bf16 \
    $MAX_FLAG 2>&1 | tee -a "$ASR_ROOT/pseudo_label.log"

log "  Pseudo-labeling done: $(wc -l < "$PSEUDO_RAW") total entries"

# -----------------------------------------------------------------------------
# Step 2: Filter pseudo-labels (kept for training)
# -----------------------------------------------------------------------------

log "=== Step 2: Filter pseudo-labels ==="
if [[ -s "$PSEUDO_FILTERED" ]]; then
    log "  $PSEUDO_FILTERED exists ($(wc -l < "$PSEUDO_FILTERED") entries) — skipping"
else
    python whisper_asr/filter_pseudo.py \
        --input "$PSEUDO_RAW" \
        --output "${PSEUDO_FILTERED%.jsonl}" \
        --confidence_threshold "$CONF_THRESH" \
        --repetition_threshold "$REP_THRESH" \
        --sw_confidence_threshold "$SW_LID_THRESH" \
        2>&1 | tee -a "$ASR_ROOT/filter.log"
fi

# -----------------------------------------------------------------------------
# Step 3: Build the "gold-where-available" variant (upper bound)
# -----------------------------------------------------------------------------

log "=== Step 3: Gold-where-available variant ==="
if [[ -s "$PSEUDO_GOLD" ]]; then
    log "  $PSEUDO_GOLD exists ($(wc -l < "$PSEUDO_GOLD") entries) — skipping"
else
    python whisper_asr/filter_pseudo.py \
        --input "$PSEUDO_RAW" \
        --output "${PSEUDO_GOLD%.jsonl}" \
        --confidence_threshold "$CONF_THRESH" \
        --repetition_threshold "$REP_THRESH" \
        --sw_confidence_threshold "$SW_LID_THRESH" \
        --use_gold_when_available \
        2>&1 | tee -a "$ASR_ROOT/filter_gold.log"
fi

# -----------------------------------------------------------------------------
# Step 4: Train 4 variants
# -----------------------------------------------------------------------------

train_variant() {
    local NAME=$1
    local PSEUDO_FLAG=$2
    local OUT="$ASR_ROOT/$NAME"

    log "--- Training: $NAME ---"
    if [[ -f "$OUT/final/config.json" ]]; then
        log "    Already trained at $OUT/final — skipping"
        return 0
    fi

    $LAUNCH whisper_asr/train.py \
        --base_model "$STUDENT_MODEL" \
        --kenspeech_dir "$KENSPEECH_DIR" \
        --output_dir "$OUT" \
        --batch_size "$BS" --grad_accum "$GA" \
        --lr "$LR" --epochs "$EPOCHS" --warmup_steps "$WARMUP" \
        --precision "$PRECISION" \
        --num_workers 4 --seed "$SEED" \
        $PSEUDO_FLAG 2>&1 | tee "$OUT.log"

    log "    Done: $NAME"
}

log "=== Step 4: Training (4 variants) ==="
train_variant "ft_kenspeech_only"             ""
train_variant "ft_kenspeech_pseudo_raw"       "--pseudo_labels_path $PSEUDO_RAW"
train_variant "ft_kenspeech_pseudo_filtered"  "--pseudo_labels_path $PSEUDO_FILTERED"
train_variant "ft_kenspeech_gold_upper_bound" "--pseudo_labels_path $PSEUDO_GOLD"

# -----------------------------------------------------------------------------
# Step 5: WER eval (vanilla baseline + 4 trained variants)
# -----------------------------------------------------------------------------

log "=== Step 5: WER eval on FLEURS sw_ke ==="

MODEL_ARGS="vanilla_small=$STUDENT_MODEL"
for NAME in ft_kenspeech_only ft_kenspeech_pseudo_raw ft_kenspeech_pseudo_filtered ft_kenspeech_gold_upper_bound; do
    if [[ -d "$ASR_ROOT/$NAME/final" ]]; then
        MODEL_ARGS="$MODEL_ARGS $NAME=$ASR_ROOT/$NAME/final"
    else
        log "  Skipping $NAME (no final/ dir)"
    fi
done

python whisper_asr/eval_wer.py \
    --models $MODEL_ARGS \
    --audio_dir "$FLEURS_TEST_DIR/audio" \
    --references_path "$FLEURS_TEST_DIR/refs.jsonl" \
    --output_path "$ASR_ROOT/results.txt" \
    --show_examples 12 \
    2>&1 | tee -a "$ASR_ROOT/eval.log"

log "=== ALL DONE at $(date) ==="
log "Results: $ASR_ROOT/results.txt"
log "Disk usage:"
du -sh "$ASR_ROOT" | tee -a "$ASR_ROOT/orchestrator.log"
