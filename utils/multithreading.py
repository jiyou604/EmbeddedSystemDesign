import cv2
import numpy as np
import time
import threading
import queue
import onnxruntime as ort
from picamera2 import Picamera2

CONF_THRESH = 0.3
IOU_THRESH = 0.45
INPUT_SIZE = 640

# 디스플레이 큐
frame_queue = queue.Queue(maxsize=1)

# 디스플레이 스레드 함수
def display_thread():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        cv2.imshow("YOLOv5n ONNX", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

# PiCamera2 설정
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (480, 480)})
picam2.configure(config)
picam2.start()
print("PiCamera started")

# ONNX 모델 로딩
onnx_model_path = "./utils/best.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("ONNX model loaded")

# 디스플레이 스레드 시작
thread = threading.Thread(target=display_thread)
thread.start()

# 추론 루프
prev_time = time.time()
while True:
    frame = picam2.capture_array()
    orig_h, orig_w = frame.shape[:2]
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
    img_input = np.expand_dims(img_input, axis=0)

    # ONNX 추론
    outputs = session.run(None, {input_name: img_input})[0]  # (1, num_boxes, 85)
    preds = np.squeeze(outputs)  # (num_boxes, 85)

    boxes = []
    for pred in preds:
        x, y, w, h = pred[0:4]
        conf = pred[4]
        cls_scores = pred[5:]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]
        score = conf * cls_conf

        if score > CONF_THRESH:
            # Convert from xywh to x1y1x2y2
            x1 = int((x - w / 2) * orig_w / INPUT_SIZE)
            y1 = int((y - h / 2) * orig_h / INPUT_SIZE)
            x2 = int((x + w / 2) * orig_w / INPUT_SIZE)
            y2 = int((y + h / 2) * orig_h / INPUT_SIZE)
            boxes.append((x1, y1, x2, y2, score, cls_id))

    # 시각화
    result_frame = frame.copy()
    for (x1, y1, x2, y2, score, cls_id) in boxes:
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {cls_id} {score:.2f}"
        cv2.putText(result_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # FPS 계산
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    print(f"FPS: {fps:.2f}")

    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 디스플레이
    if not frame_queue.full():
        frame_queue.put(result_frame)

# 종료 처리
picam2.stop()
frame_queue.put(None)
thread.join()
