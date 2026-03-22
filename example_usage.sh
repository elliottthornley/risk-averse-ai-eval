#!/bin/bash
# Example usage script for risk-averse AI evaluation
#
# This script shows how to evaluate a fine-tuned model on key datasets.
# Modify the MODEL_PATH and BASE_MODEL variables for your setup.

MODEL_PATH="/path/to/your/model/adapter"  # CHANGE THIS
BASE_MODEL="Qwen/Qwen3-8B"                # CHANGE THIS if needed

echo "==================================="
echo "Risk-Averse AI Evaluation Examples"
echo "==================================="
echo ""

# Example 1: High-Stakes Test (Primary held-out test set)
echo "1. Evaluating on high-stakes OOD test set..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset high_stakes_test \
    --num_situations 50 \
    --temperature 0 \
    --output results_high_stakes_test.json

echo ""
echo "Results saved to: results_high_stakes_test.json"
echo ""

# Example 2: Astronomical Deployment Set
echo "2. Evaluating on astronomical-stakes deployment set..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset astronomical_stakes_deployment \
    --num_situations 50 \
    --temperature 0 \
    --output results_astronomical_stakes_deployment.json

echo ""
echo "Results saved to: results_astronomical_stakes_deployment.json"
echo ""

# Example 3: OOD Validation (for continuity with earlier experiments)
echo "3. Evaluating on OOD validation set..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset ood_validation \
    --num_situations 50 \
    --temperature 0 \
    --output results_ood_validation.json

echo ""
echo "Results saved to: results_ood_validation.json"
echo ""

# Example 4: In-Distribution Validation Slice
echo "4. Evaluating on an in-distribution validation slice..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset indist_validation \
    --start_position 901 \
    --end_position 1000 \
    --num_situations 100 \
    --temperature 0 \
    --output results_indist_val.json

echo ""
echo "Results saved to: results_indist_val.json"
echo ""

# Example 5: Training Set (Overfitting Check)
echo "5. Evaluating on training set (overfitting check)..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset training \
    --num_situations 50 \
    --temperature 0 \
    --output results_train.json

echo ""
echo "Results saved to: results_train.json"
echo ""

# Summary
echo "==================================="
echo "Evaluation Complete!"
echo "==================================="
echo ""
echo "View results with:"
echo "  cat results_high_stakes_test.json | python -m json.tool | head -30"
echo ""
echo "Quick summary:"
python -c "
import json
import sys

files = [
    'results_high_stakes_test.json',
    'results_astronomical_stakes_deployment.json',
    'results_ood_validation.json',
    'results_indist_val.json',
    'results_train.json'
]
labels = [
    'High-Stakes Test',
    'Astronomical Stakes',
    'OOD Validation',
    'In-Dist Validation',
    'Training Set'
]

print('')
print('Dataset                  | CARA Rate | Parse Rate | Cooperate% | Rebel%  | Steal%')
print('-' * 90)

for fname, label in zip(files, labels):
    try:
        with open(fname) as f:
            data = json.load(f)
            cara = data['metrics']['best_cara_rate'] * 100
            parse = data['metrics']['parse_rate'] * 100
            coop = data['metrics']['cooperate_rate'] * 100
            rebel = data['metrics'].get('rebel_rate', 0) * 100
            steal = data['metrics'].get('steal_rate', 0) * 100
            print(f'{label:24} | {cara:5.1f}%    | {parse:6.1f}%    | {coop:6.1f}%    | {rebel:5.1f}% | {steal:5.1f}%')
    except FileNotFoundError:
        print(f'{label:24} | ERROR: File not found')
    except Exception as e:
        print(f'{label:24} | ERROR: {e}')

print('')
"

echo ""
echo "Done!"
