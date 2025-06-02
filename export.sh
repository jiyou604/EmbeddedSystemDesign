#!/bin/bash

# ===============================
# ONNX 모델 단순화 및 양자화 스크립트
# ===============================

# 사용자 지정 경로 설정
INPUT_MODEL="./custom_model/augmented/exp/weights/best.onnx"
SIMPLIFIED_MODEL="./custom_model/augmented/exp/weights/best_simplified.onnx"
OUTPUT_MODEL="./custom_model/augmented/exp/weights/best_quantized_int8.onnx"

# === Step 1: 모델 단순화 (ONNX Simplifier 사용) ===
echo "[1/3] Simplifying the model using onnxsim..."
python3 -m onnxsim "$INPUT_MODEL" "$SIMPLIFIED_MODEL"
if [ $? -ne 0 ]; then
    echo "❌ Failed to simplify the ONNX model."
    exit 1
fi
echo "✅ Model simplified: $SIMPLIFIED_MODEL"

# === Step 2: 동적 양자화 (Dynamic Quantization) ===
echo "[2/3] Applying dynamic quantization using onnxruntime..."
python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input='$SIMPLIFIED_MODEL',
    model_output='$OUTPUT_MODEL',
    weight_type=QuantType.QInt8
)
"
if [ $? -ne 0 ]; then
    echo "❌ Dynamic quantization failed."
    exit 1
fi
echo "✅ Quantized model saved as: $OUTPUT_MODEL"

# === Step 3: 임시 단순화 파일 삭제 (선택 사항) ===
echo "[3/3] Cleaning up temporary simplified model..."
rm "$SIMPLIFIED_MODEL"
echo "🧹 Cleaned up: $SIMPLIFIED_MODEL"

# === 완료 메시지 ===
echo "🎉 Done! Final quantized model: $OUTPUT_MODEL"
