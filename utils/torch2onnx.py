import torch

# PyTorch 모델 로드
model_path = '../custom_model/PB/exp2/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

# ONNX 변환용 더미 입력 (YOLO는 640x640 기본)
dummy_input = torch.randn(1, 3, 480, 480)

# ONNX로 내보내기
torch.onnx.export(
    model,
    dummy_input,
    "yolov5_custom.onnx",
    input_names=['images'],
    output_names=['output'],
    opset_version=12,
    dynamic_axes={
        'images': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ ONNX 모델 변환 완료: yolov5_custom_480x480.onnx")
