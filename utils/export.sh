python ./yolov5/export.py --weights ./custom_model/PB/exp13/weights/best.pt --include torchscript onnx

python -m onnxsim ./custom_model/PB/exp13/weights/best.onnx best_simplified.onnx
