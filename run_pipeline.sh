#!/bin/bash
set -e

DATASET=$1
LOG="pipeline_${DATASET}.log"

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_pipeline.sh <DATASET>"
    exit 1
fi

echo "Starting pipeline for $DATASET..."

CUDA_VISIBLE_DEVICES=0 python debug/learn_dnn_misc_detector.py
for dir in tmp checkpoints lightning_logs trained_models
do
    if [ -d "$dir" ]; then
        mv "$dir" debug/
    fi
done
python generate_SCC_outputs.py
python build_ensembles.py --dataset "$DATASET" --fcc_alrs 0.01 0.001 0.0001 --ensemble_sizes 2 3 4
python build_ensembles_sorted_accuracy.py --dataset "$DATASET"  --fcc_alrs 0.01 0.001 0.0001 --ensemble_sizes 2 3 4
python compute_classifier_scc_stats.py
python run_scc_couples.py debug/SCC_outputs

# Create dataset folder inside debug
mkdir -p "debug/$DATASET"

# Move output folders into debug/$DATASET
for dir in checkpoints lightning_logs SCC_outputs tmp trained_models ENSEMBLE_outputs SCC_couples ENSEMBLE_outputs_accuracy_sorted
do
    if [ -d "debug/$dir" ]; then
        mv "debug/$dir" "debug/$DATASET/"
    fi
done

echo "Pipeline finished for $DATASET."

#remmeber to change dataset path in the learn_dnn_misc_detector