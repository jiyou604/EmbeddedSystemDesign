import cv2
import time
import onnxruntime as ort
import numpy as np
from picamera2 import Picamera2

# ONNX load
model_path='/home/pi/ESD/EmbeddedSystemDesign/custom_model/augmented/exp/weights/best_quantized.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Picam
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

def preprocess(img):
    img_resized = cv2.resize(img, (480, 480))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_tensor, axis=0)

def postprocess(outputs, orig_shape, conf_thresh=0.4):
    boxes = []
    confidences = []
    class_ids = []

    predictions = outputs[0][0]  # (num_boxes, 85)

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
        x = int((cx - w/2) * orig_shape[1] / 480)
        y = int((cy - h/2) * orig_shape[0] / 480)
        w = int(w * orig_shape[1] / 480)
        h = int(h * orig_shape[0] / 480)

        boxes.append([x, y, w, h])
        confidences.append(float(conf))
        class_ids.append(int(class_id))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.5)
    result = []
    for i in indices:
        #i = i[0]
        result.append((boxes[i], confidences[i], class_ids[i]))
    return result


def draw_boxes(frame, results):
    for (box, score, class_id) in results:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"ID:{class_id} {score:.2f}"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


prev_time = time.time()

while True:
    frame = picam2.capture_array()

    input_tensor = preprocess(frame)
    outputs = session.run(None, {'images': input_tensor})

    results = postprocess(outputs, frame.shape)
    frame = draw_boxes(frame, results)

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    print(f"FPS: {fps:.2f}")

    cv2.imshow("ONNX YOLOv5 - PiCam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
