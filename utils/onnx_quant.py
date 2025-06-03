import cv2
import time
import onnxruntime as ort
import numpy as np
from picamera2 import Picamera2

# parameters
resolution = 320

# ONNX load
model_path='/home/pi/ESD/EmbeddedSystemDesign/custom_model/augmented/exp/weights/best_fp16.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Picam
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (resolution, resolution)})
picam2.configure(config)
picam2.start()

def preprocess(img):
    img_resized = cv2.resize(img, (320, 320))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float16) / 255.0
    
    return np.expand_dims(img_tensor, axis=0)  # (1, 3, 480, 480)

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
        x = int((cx - w/2) * orig_shape[1] / resolution)
        y = int((cy - h/2) * orig_shape[0] / resolution)
        w = int(w * orig_shape[1] / resolution)
        h = int(h * orig_shape[0] / resolution)

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
    start_time = time.time()
    input_tensor = preprocess(frame)
    preprocess_time = time.time()
    print(f"preprocess: {preprocess_time-start_time}")
    outputs = session.run(None, {'images': input_tensor})
    inference_time = time.time()
    print(f"inference: {inference_time-preprocess_time}")

    results = postprocess(outputs, frame.shape)
    postprocess_time = time.time()
    print(f"postprocess: {postprocess_time-inference_time}")
    
    frame = draw_boxes(frame, results)
    boxing_time = time.time()
    print(f"box: {boxing_time-postprocess_time}")

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # print(f"FPS: {fps:.2f}")
    cal_time = time.time()
    print(f"cal: {cal_time-boxing_time}")

    cv2.imshow("ONNX YOLOv5 - PiCam", frame)
    plot_time = time.time()
    print(f"plot: {plot_time-cal_time}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
