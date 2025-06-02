import cv2
import time
import onnxruntime as ort
import numpy as np
import threading
import queue
from picamera2 import Picamera2

# ONNX load
model_path = '/home/pi/ESD/EmbeddedSystemDesign/custom_model/augmented/exp/weights/best_simplified_quant.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (480, 480)})
picam2.configure(config)
picam2.start()

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def preprocess(img):
    # img_resized = cv2.resize(img, (480, 480))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_tensor, axis=0)

def postprocess(outputs, orig_shape, conf_thresh=0.4):
    boxes, confidences, class_ids = [], [], []

    predictions = outputs[0][0]
    for pred in predictions:
        obj_conf = pred[4]
        if obj_conf < conf_thresh:
            continue

        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        conf = obj_conf * class_scores[class_id]

        if conf < conf_thresh:
            continue

        cx, cy, w, h = pred[0:4]
        x = int((cx - w / 2) * orig_shape[1] / 480)
        y = int((cy - h / 2) * orig_shape[0] / 480)
        w = int(w * orig_shape[1] / 480)
        h = int(h * orig_shape[0] / 480)

        boxes.append([x, y, w, h])
        confidences.append(float(conf))
        class_ids.append(int(class_id))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.5)
    return [(boxes[i], confidences[i], class_ids[i]) for i in indices]

def draw_boxes(frame, results):
    for (box, score, class_id) in results:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID:{class_id} {score:.2f}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

# Inference
def inference_thread():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        input_tensor = preprocess(frame)
        outputs = session.run(None, {'images': input_tensor})
        results = postprocess(outputs, frame.shape)
        result_queue.put((frame, results))

# Display
def display_thread():
    prev_time = time.time()
    while True:
        try:
            frame, results = result_queue.get(timeout=1)
        except queue.Empty:
            continue
        frame = draw_boxes(frame, results)
        fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()
        print(f"FPS: {fps:.2f}")
        cv2.imshow("ONNX YOLOv5 - PiCam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Multithreading
t1 = threading.Thread(target=inference_thread, daemon=True)
t2 = threading.Thread(target=display_thread, daemon=True)
t1.start()
t2.start()

try:
    while True:
        frame = picam2.capture_array()
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    frame_queue.put(None)
    t1.join()
    print("Thread stopped.")
