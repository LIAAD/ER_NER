#!/bin/bash
# ============================================================
# Fine-tune MedAlBERTina for Portuguese Clinical NER
# Usage: bash run_train.sh
# ============================================================

set -e  # Exit immediately on error

# ---------- Paths ----------
TRAIN_JSON="train.json"
VAL_JSON="val.json"
OUTPUT_DIR="output-medialbertina-pt-pt-900m-realtest"  # where the best model will be saved

# ---------- Model ----------
MODEL="portugueseNLP/medialbertina_pt-pt_900m"
#MODEL="pucpr/biobertpt-all"



# ---------- Hyperparameters ----------
EPOCHS=20
BATCH_SIZE=4
GRAD_ACCUM=4          # Effective batch size = BATCH_SIZE * GRAD_ACCUM = 16
MAX_LEN=512
STRIDE=128            # bumped from 96 → 128 for better long-doc coverage
LR=2e-5
PRECISION="bf16"      # bf16 | fp16 | fp32

# ---------- Early Stopping ----------
EARLY_STOPPING=true
PATIENCE=3

# ---------- Class-imbalance handling ----------
# Each training doc containing an "Alergias medicamentosas__Negativa" span
# is duplicated this many times (1 = no oversampling).
# 4-5 is the recommended starting point for the current dataset.
OVERSAMPLE_NEGATIVA=4

# ---------- GPU (leave empty to use all available) ----------
GPU="2"

# ============================================================

echo "============================================"
echo "  Portuguese Clinical NER — Training Start"
echo "============================================"
echo "Model        : $MODEL"
echo "Train JSON   : $TRAIN_JSON"
echo "Val   JSON   : $VAL_JSON"
echo "Output dir   : $OUTPUT_DIR"
echo "Epochs       : $EPOCHS"
echo "Batch size   : $BATCH_SIZE (grad_accum=$GRAD_ACCUM → effective=$(($BATCH_SIZE * $GRAD_ACCUM)))"
echo "Max len      : $MAX_LEN  |  Stride: $STRIDE"
echo "LR           : $LR"
echo "Precision    : $PRECISION"
echo "Oversample N : x$OVERSAMPLE_NEGATIVA"
echo "GPU          : $GPU"
echo "============================================"

# Sanity-check the data paths exist before launching a long job.
for f in "$TRAIN_JSON" "$VAL_JSON"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: data file not found: $f"
        exit 1
    fi
done

ES_FLAG=""
if [ "$EARLY_STOPPING" = true ]; then
    ES_FLAG="--early_stopping --patience $PATIENCE"
fi

GPU_FLAG=""
if [ -n "$GPU" ]; then
    GPU_FLAG="--gpu $GPU"
fi

export CUDA_VISIBLE_DEVICES=$GPU
python train.py \
    --train_json  "$TRAIN_JSON" \
    --val_json    "$VAL_JSON" \
    --model       "$MODEL" \
    --output_dir  "$OUTPUT_DIR" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --grad_accum  "$GRAD_ACCUM" \
    --max_len     "$MAX_LEN" \
    --stride      "$STRIDE" \
    --lr          "$LR" \
    --precision   "$PRECISION" \
    --oversample_negativa "$OVERSAMPLE_NEGATIVA" \
    $ES_FLAG \
    $GPU_FLAG

# ---------- Post-training cleanup ----------
# Trainer's transient `checkpoint-XXX/` folder duplicates the model that the
# Python script has already saved to `${OUTPUT_DIR}/best/`. Remove it so only
# the best epoch's model remains on disk (~1.8 GB saved for a 900M-param model).
echo ""
echo "[Cleanup] Removing transient Trainer checkpoint folders..."
if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    rm -rf "${OUTPUT_DIR}"/checkpoint-*
    echo "[Cleanup] Done. Kept only ${OUTPUT_DIR}/best/"
else
    echo "[Cleanup] No transient checkpoints found (nothing to remove)."
fi

echo ""
echo "============================================"
echo "  Training complete! Model saved to: $OUTPUT_DIR/best"
echo "============================================"