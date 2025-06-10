import cv2
import time
import os
import onnxruntime as ort
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from control import Stepper
from control.PID import PID
from Astar import Pathfinder

os.environ["OMP_NUM_THREADS"] = "4"

# ONNX load
model_path='/home/pi/ESD/EmbeddedSystemDesign/custom_model/augmented/exp/weights/best_fp16.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 1
session_options.enable_mem_pattern = False
session_options.enable_cpu_mem_arena = False

cv2.setNumThreads(4)

# parameters
model_size = 320
resolution = 640
destination = [0, 0]
Kp, Ki, Kd = [0.1, 0.001, 0.37]
# Kp, Ki, Kd = [0.09, 0.0, 0.11]
# Kp, Ki, Kd = [0.09, 0.0, 0.11]
# Kp, Ki, Kd = [0.033, 0.0, 0.08]
max_rotation = 500
toggle_threshold = 10

pid_x = PID(kp=Kp, ki=Ki, kd=Kd)
pid_y = PID(kp=Kp, ki=Ki, kd=Kd)

# motor classes
GPIO.setmode(GPIO.BCM)
motor_x0 = Stepper.Motor([14, 15, 17, 18], max_step=max_rotation)
motor_x1 = Stepper.Motor([27, 22, 23, 24], max_step=max_rotation)
motor_y0 = Stepper.Motor([10, 9, 25, 11], max_step=max_rotation)
motor_y1 = Stepper.Motor([16, 26, 20, 21], max_step=max_rotation)

platform = Stepper.Platform([motor_x0, motor_x1, motor_y0, motor_y1])

# Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (resolution, resolution)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def preprocess(img):
    img_resized = cv2.resize(img, (model_size, model_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float16) / 255.0
    
    return np.expand_dims(img_tensor, axis=0)  # (1, 3, 480, 480)

def postprocess(outputs, orig_shape, conf_thresh=0.35):
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
        x = int((cx - w/2) * orig_shape[1] / model_size)
        y = int((cy - h/2) * orig_shape[0] / model_size)
        w = int(w * orig_shape[1] / model_size)
        h = int(h * orig_shape[0] / model_size)

        boxes.append([x, y, w, h])
        confidences.append(float(conf))
        class_ids.append(int(class_id))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.5)
    result = []
    for i in indices:
        #i = i[0]
        result.append((boxes[i], confidences[i], class_ids[i]))
    return result

def draw_boxes_and_center(frame, results):
    if not results:
        return frame, None, None
    box, score, class_id = results[0]
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"ID:{class_id} {score:.2f}"
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    center_x = x + w // 2
    center_y = y + h // 2
    return frame, center_x, center_y

prev_time = time.time()

try:
    path = None
    current_node = 0
    last_steps = [0, 0]

    while True:
        frame = picam2.capture_array()
        input_tensor = preprocess(frame)
        outputs = session.run(None, {'images': input_tensor})
        results = postprocess(outputs, frame.shape)
        frame, curr_x, curr_y = draw_boxes_and_center(frame, results)

        if len(results) == 0 or curr_x is None or curr_y is None:
            print("PB not detected. Waiting...")
            cv2.imshow("ONNX YOLOv5 - PiCam", frame)
            steps_x = int(last_steps[0] * -0.5)
            steps_y = int(last_steps[1] * -0.5)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if path is None:
            # PB 위치 기반 A* 경로 탐색
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray_img.shape[:2]
            grid = gray_img.copy()

            for (box, _, _) in results:
                x, y, w, h = box
                grid[y:y+h, x:x+w] = 255

            start = (curr_y, curr_x)
            goal = (0, grid.shape[1]-1)

            finder = Pathfinder(grid, start, goal)
            path = finder.get_path()
            current_node = 0

            if not path:
                print("No path found. Waiting...")
                path = None
                continue

            print(f"Path found: {path}")

        # 경로 따라 이동
        if current_node < len(path):
            if path[current_node] is not None:
                target_y, target_x = path[current_node]

                pid_x.setpoint = target_x
                pid_y.setpoint = target_y

                x_output = pid_x.compute(curr_x - target_x)
                y_output = pid_y.compute(curr_y - target_y)

                x_steps = int(-x_output)
                y_steps = int(y_output)

                platform.tilt(x_steps, y_steps)

                dist = ((curr_x - target_x)**2 + (curr_y - target_y)**2) ** 0.5
                if dist < 3:
                    current_node += 1
                    print(f"Reached node {current_node}/{len(path)}")

                

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        print(f"FPS: {fps:.2f}")
        cv2.imshow("ONNX YOLOv5 - PiCam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    picam2.stop()
    cv2.destroyAllWindows()
    platform.cleanup()
