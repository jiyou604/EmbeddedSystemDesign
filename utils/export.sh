python ./yolov5/export.py --weights ./custom_model/augmented/exp/weights/best.pt --img 480 --include torchscript onnx

python -m onnxsim ./custom_model/augmented/exp/weights/best.onnx ./custom_model/augmented/exp/weights/best_simplified.onnx
