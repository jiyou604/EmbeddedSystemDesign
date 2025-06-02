#!/bin/bash

#Parameters
INPUT_PATH="./custom_model/augmented/exp/weights/best.pt"
ONNX_PATH="./custom_model/augmented/exp/weights/best.onnx"
SIMP_PATH="./custom_model/augmented/exp/weights/best_simplified.onnx"

# === Step 1: Export to ONNX ===
echo "Exporting yolov5 to ONNX..."
python ./yolov5/export.py --weights "$INPUT_PATH" --img 480 --include torchscript onnx #--img 680

# === Step 2: Simplify the model ===
echo "Simplifying the model..."
python3 -m onnxsim "$ONNX_PATH" "$SIMP_PATH"

# === Step 3: Dynamic Quantization ===
echo "Applying dynamic quantization..."
python3 ./utils/quantize.py

