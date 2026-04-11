#!/bin/bash

# run_experiments.sh — Sequentially execute all music genre classification experiments.

# Always run from the project root (one level up from scripts/)
cd "$(dirname "$0")/.."

# List of experiment configs in configs/
EXPERIMENTS=(
    "rnn_3s"
    "lstm_3s"
    "cnn2d_3s"
    "cnn2d_3c_3s"
    "crnn_3s"
    "crnn_6s"
    "crnna_3s"
    "crnn3c_3s"
    "crnn3ca_3s"
)

LOG_DIR="outputs/logs"
FINAL_REPORT="outputs/final_summary.txt"

echo "=================================================="
echo " STARTING SEQUENTIAL EXPERIMENT RUNS "
echo " Time: $(date)"
echo "=================================================="

# 1. Pre-run validation: Check if all experiment configs exist
echo "Validating experiment configurations..."
for EXP in "${EXPERIMENTS[@]}"; do
    CONFIG_PATH="configs/$EXP.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Configuration file '$CONFIG_PATH' not found!"
        exit 1
    fi
done
echo "All configurations validated successfully."
echo ""

# 2. Loop through each experiment and execute the training script
for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo " RUNNING EXPERIMENT: $EXP "
    echo "--------------------------------------------------"
    
    # Execute training
    # We do NOT use set -e, so we can check the exit status manually
    python scripts/train_kfold.py --exp "$EXP"
    
    if [ $? -eq 0 ]; then
        echo "Successfully finished experiment: $EXP"
    else
        echo "Warning: Experiment $EXP failed. Continuing to the next one..."
    fi
done

echo ""
echo "=================================================="
echo " GENERATING FINAL SUMMARY REPORT "
echo "=================================================="

# Initialize the report file
echo "Music Genre Classification - Final Summary Report" > "$FINAL_REPORT"
echo "Generated on: $(date)" >> "$FINAL_REPORT"
echo "--------------------------------------------------" >> "$FINAL_REPORT"
printf "%-20s | %-12s | %-12s | %-12s\n" "Experiment" "Avg Val Acc" "Avg Test Acc" "Ensemble Acc" >> "$FINAL_REPORT"
echo "----------------------------------------------------------------------" >> "$FINAL_REPORT"

# Parse each results.yaml to collect metrics
for EXP in "${EXPERIMENTS[@]}"; do
    RESULTS_FILE="$LOG_DIR/$EXP/results.yaml"
    
    if [ -f "$RESULTS_FILE" ]; then
        # Use grep and awk to extract the accuracy values from YAML
        VAL_ACC=$(grep "avg_val_acc" "$RESULTS_FILE" | awk '{print $2}')
        TEST_ACC=$(grep "avg_test_acc" "$RESULTS_FILE" | awk '{print $2}')
        ENS_ACC=$(grep "ensemble_test_acc" "$RESULTS_FILE" | awk '{print $2}')
        
        # Format Val Acc
        if [ -z "$VAL_ACC" ]; then VAL_ACC="N/A"; else VAL_ACC=$(printf "%.4f" "$VAL_ACC"); fi
        # Format Test Acc
        if [ -z "$TEST_ACC" ]; then TEST_ACC="N/A"; else TEST_ACC=$(printf "%.4f" "$TEST_ACC"); fi
        # Format Ensemble Acc
        if [ -z "$ENS_ACC" ]; then ENS_ACC="N/A"; else ENS_ACC=$(printf "%.4f" "$ENS_ACC"); fi
        
        printf "%-20s | %-12s | %-12s | %-12s\n" "$EXP" "$VAL_ACC" "$TEST_ACC" "$ENS_ACC" >> "$FINAL_REPORT"
    else
        printf "%-20s | %-12s | %-12s | %-12s\n" "$EXP" "Failed/N/A" "N/A" "N/A" >> "$FINAL_REPORT"
    fi
done

echo "--------------------------------------------------" >> "$FINAL_REPORT"

# Print the final report to the terminal
cat "$FINAL_REPORT"

# Scan for any missing confusion matrices
echo ""
python scripts/evaluate_confusion.py

echo ""
echo "Full report saved to: $FINAL_REPORT"
echo "=================================================="
echo " PIPELINE COMPLETED "
echo " Time: $(date)"
echo "=================================================="
