import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess


preprocessed_path = "./custom_model/augmented/exp/weights/best_simplified_preprocessed.onnx"
fp16_model_path = "./custom_model/augmented/exp/weights/best_fp16.onnx"

model = onnx.load(preprocessed_path)

model_fp16 = float16.convert_float_to_float16(model)

onnx.save(model_fp16, fp16_model_path)