#!/bin/bash

MODEL_PATH="/path/to/your/model/adapter"  # CHANGE THIS
BASE_MODEL="Qwen/Qwen3-8B"                # CHANGE THIS if needed

echo "==================================="
echo "Risk-Averse AI Evaluation Examples"
echo "==================================="
echo ""

echo "1. Medium-stakes validation (recommended 200-situation run)..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset medium_stakes_validation \
    --num_situations 200 \
    --output results_medium_validation.json

echo ""
echo "2. High-stakes test..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset high_stakes_test \
    --num_situations 1000 \
    --output results_high_stakes_test.json

echo ""
echo "3. Astronomical-stakes deployment..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset astronomical_stakes_deployment \
    --num_situations 1000 \
    --output results_astronomical_stakes_deployment.json

echo ""
echo "4. Steals-only test..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset steals_test \
    --num_situations 1000 \
    --output results_steals_test.json

echo ""
echo "5. Low-stakes training set..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset low_stakes_training \
    --num_situations 200 \
    --output results_low_stakes_training.json

echo ""
echo "6. Held-out low-stakes slice from the same source CSV..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset low_stakes_training \
    --start_position 901 \
    --end_position 1000 \
    --num_situations 100 \
    --output results_low_stakes_validation_slice.json

echo ""
echo "7. LIN-only low-stakes run..."
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --dataset low_stakes_training \
    --lin_only \
    --num_situations 200 \
    --output results_low_stakes_lin_only.json

echo ""
echo "Always save responses. Do not use --no_save_responses unless you have already talked to Elliott."
