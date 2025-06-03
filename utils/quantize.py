import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess


fp32_model_path = "./custom_model/augmented/exp/weights/best_simplified.onnx"
preprocessed_path = "./custom_model/augmented/exp/weights/best_preprocessed.onnx"
fp16_model_path = "./custom_model/augmented/exp/weights/best_fp16.onnx"
output_path="./custom_model/augmented/exp/weights/best_quantized.onnx"

model = onnx.load(fp32_model_path)

model_fp16 = float16.convert_float_to_float16(model)

onnx.save(model_fp16, fp16_model_path)

quantize_dynamic(
    model_input=fp16_model_path,
    model_output=output_path,
    weight_type=QuantType.QInt8,
)
