import cv2
import numpy as np
import time
import threading
import queue
import onnxruntime as ort
from picamera2 import Picamera2

confidence = 0.5
IOU = 0.3
class_names = ["PB"] 
frame_queue = queue.Queue(maxsize=1)

def display_thread():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        cv2.imshow("YOLOv5n ONNX", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

# NMS
def non_max_suppression(prediction, conf_thres=confidence, iou_thres=IOU):
    boxes = []
    confidences = []
    confidences = []
    confidences = []
    class_ids = []

    for det in prediction[0]:  # shape: [num_detections, 6]
        conf = det[4]
        if conf > conf_thres:
            scores = det[5:]
            class_id = np.argmax(scores)
            score = scores[class_id] * conf
            if score > conf_thres:
                x1, y1, x2, y2 = det[0:4]
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(score))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)
    results = []
    for i in indices:
        results.append((boxes[i], confidences[i], class_ids[i]))
    return results

# PiCam
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 640)})
picam2.configure(config)
picam2.start()
print("PiCamera started")

# ONNX load
onnx_model_path = "/home/pi/ESD/EmbeddedSystemDesign/best_simplified.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("ONNX model loaded")

# Display threading
thread = threading.Thread(target=display_thread)
thread.start()

# inference
prev_time = time.time()
while True:
    frame = picam2.capture_array()
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
    img_input = np.expand_dims(img_input, axis=0)  # [1, 3, 480, 480]
    img_input = np.ascontiguousarray(img_input)

    # ONNX Inference
    outputs = session.run(None, {input_name: img_input})[0]

    # Postprocessing
    detections = non_max_suppression(outputs, conf_thres=confidence, iou_thres=IOU)

    # Visulaization
    for box, conf, cls_id in detections:
        x, y, w, h = map(int, box)
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display Queue
    if not frame_queue.full():
        frame_queue.put(frame.copy())


picam2.stop()
frame_queue.put(None)
thread.join()
