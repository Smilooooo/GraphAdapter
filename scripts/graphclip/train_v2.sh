#!/bin/bash
# =============================================================================
# GraphCLIP v2 Training Script - Multi-backbone Support
# =============================================================================
# 
# Usage:
#   ./scripts/graphclip/train_v2.sh <DATASET> <BACKBONE> <SHOTS> [SEED]
#
# Examples:
#   ./scripts/graphclip/train_v2.sh bracs biomedclip 16
#   ./scripts/graphclip/train_v2.sh lunghist700 plip 4 42
#   ./scripts/graphclip/train_v2.sh iciar2018 quiltnet 8
#
# Available backbones: rn50, biomedclip, plip, quiltnet, conch
# Available datasets: bracs, lunghist700, iciar2018, caltech101, eurosat, etc.
# =============================================================================

DATA=/data/datasets/
TRAINER=GraphCLIP_v2

DATASET=$1       # e.g., bracs, lunghist700
BACKBONE=$2      # e.g., biomedclip, plip, quiltnet, conch, rn50
SHOTS=${3:-16}   # Number of shots (default: 16)
SEED=${4:-1}     # Random seed (default: 1)

# Validate inputs
if [ -z "$DATASET" ] || [ -z "$BACKBONE" ]; then
    echo "Usage: $0 <DATASET> <BACKBONE> [SHOTS] [SEED]"
    echo "  DATASET:  bracs, lunghist700, iciar2018, etc."
    echo "  BACKBONE: rn50, biomedclip, plip, quiltnet, conch"
    echo "  SHOTS:    1, 2, 4, 8, 16 (default: 16)"
    echo "  SEED:     random seed (default: 1)"
    exit 1
fi

# Check if config exists
CFG_FILE="configs/trainers/${TRAINER}/${BACKBONE}.yaml"
if [ ! -f "$CFG_FILE" ]; then
    echo "Error: Config file not found: $CFG_FILE"
    echo "Available backbones:"
    ls configs/trainers/${TRAINER}/ 2>/dev/null || echo "  No configs found in configs/trainers/${TRAINER}/"
    exit 1
fi

# Output directory
DIR="output/${DATASET}/${TRAINER}/${BACKBONE}_${SHOTS}shots/seed${SEED}"

if [ -d "$DIR" ]; then
    echo "Results already exist at ${DIR}"
    echo "Delete the folder to re-run, or use a different seed."
else
    echo "=============================================="
    echo "GraphCLIP v2 Training"
    echo "=============================================="
    echo "Dataset:  $DATASET"
    echo "Backbone: $BACKBONE"
    echo "Shots:    $SHOTS"
    echo "Seed:     $SEED"
    echo "Output:   $DIR"
    echo "=============================================="
    
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${BACKBONE}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
fi
