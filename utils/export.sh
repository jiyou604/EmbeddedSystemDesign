python ./yolov5/export.py --weights ./custom_model/PB/exp13/weights/best.pt --include torchscript onnx

python -m ./custom_model/PB/exp13/weights/best.pt best_simplified.onnx
