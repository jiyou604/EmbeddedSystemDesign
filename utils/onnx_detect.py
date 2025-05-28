import cv2
import onnxruntime as ort
import numpy as np

# ONNX 세션 로드
session = ort.InferenceSession('yolov5_custom.onnx', providers=['CPUExecutionProvider'])

# 영상 파일 로드
file = './input/PB_moving.mp4'
video = cv2.VideoCapture(file)

w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = video.get(cv2.CAP_PROP_FPS)
output_path = './output/output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# 전처리 함수 (YOLOv5 포맷에 맞게 조정)
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_tensor, axis=0)

# 후처리 함수는 결과 구조에 따라 조정 필요 (클래스, bbox 정보 파싱 등)
# 여기선 생략: postprocess(output)

while True:
    ret, frame = video.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    # ONNX 추론
    outputs = session.run(None, {'images': input_tensor})  # output[0]는 raw output

    # NOTE: 실제로는 postprocessing 추가 필요 (NMS, bbox scale 등)
    # 현재는 그냥 원본 프레임 저장
    out.write(frame)
    cv2.imshow("ONNX YOLOv5 (raw)", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
