from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "./custom_model/augmented/exp/weights/best_simplified.onnx"
model_quant = "./custom_model/augmented/exp/weights/best_simplified_quant.onnx"

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QInt8  # or QuantType.QUInt8
)
