import cv2
import time
import os
import onnxruntime as ort
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from control import Stepper
from control.PID import PID
from Astar.py import Pathfinder

os.environ["OMP_NUM_THREADS"] = "4"

# ONNX load
model_path='/home/pi/ESD/EmbeddedSystemDesign/custom_model/augmented/exp/weights/best_fp16.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# parameters
model_size = 320
resolution = 640
destination = [0, 0]
Kp, Ki, Kd = [0.09, 0.0, 0.11]
# Kp, Ki, Kd = [0.09, 0.0, 0.11]
# Kp, Ki, Kd = [0.033, 0.0, 0.08]
max_rotation = 1000
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

def get_position(frame):
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    lower_rgb = np.array([0, 50, 100])
    upper_rgb = np.array([80, 150, 255])

    mask = cv2.inRange(blurred, lower_rgb, upper_rgb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * radius * radius
            circularity = area / circle_area
            if circularity > 0.45:
                center = (int(x), int(y))
                radius = int(radius)
                return center, radius

    return None

def preprocess(img):
    img_resized = cv2.resize(img, (model_size, model_size))
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

def draw_boxes(frame, results):
    for (box, score, class_id) in results:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"ID:{class_id} {score:.2f}"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)       
        pb_start_y = x+w//2
        pb_start_x = y+h//2 
    return frame, pb_start_y, pb_start_x

prev_time = time.time()

try:
    fframe = picam2.capture_array() ## == first frame

    # YOLO: detect starting position of PB
    input_tensor = preprocess(fframe)
    outputs = session.run(None, {'images': input_tensor})
    results = postprocess(outputs, fframe.shape)

    gray_img = cv2.cvtColor(fframe, cv2.COLOR_BGR2GRAY) ## directly return in e2e.py

    H, W = gray_img.shape[:2]
    grid = np.ones((H, W), dtype=np.uint8)
    
    fframe, pb_start_y, pb_start_x = draw_boxes(fframe, results)
    grid[y:y+h, x:x+w] = 0
    ## in gird, we mask 0 as obstacles (=dots)
    
    start = (pb_start_y, pb_start_x)    
    goal = (0, grid.shape[1] - 1)
    # Astar: find path to the target (target should be determined manually)

    finder = Pathfinder(grid, start, goal)
    path = finder.get_path()
    ## after find path, node setting = 0
    current_node = 0

    while current_node < len(path):
        frame = picam2.capture_array()
        
        input_tensor = preprocess(frame)
        outputs = session.run(None, {'images': input_tensor})
        results = postprocess(outputs, frame.shape)
        frame, pb_start_y, pb_start_x = draw_boxes(frame, results)
        
        if len(results) == 0:
            ## not detect PB
            print("not PB")
            continue

        target_y, target_x = path[current_node]

        # PID: get optimal steps to the target point
        pid_x.setpoint = target_x
        pid_y.setpoint = target_y

        x_output = pid_x.compute(pb_start_x)
        y_output = pid_y.compute(pb_start_y)
        
        pid_position = get_position(results) # get position function only finds the position of ball/PB depending on the RGB values it is not pid position
        
        # details in control/bal.py and control/balance.py
        # make PB to move following the path
        # 
        # move path[0] -> path[1] 
        # if done, move path[1] -> path [2]
        # if done, move path[2] -> path [3]
        # ... and so on
        # this might need another loop

        # Control: moving the motors based on the values from PID
        x_steps = int(-x_output)
        y_steps = int(y_output)
        platform.tilt(x_steps, y_steps)
        
        dist = ((pb_start_x - target_x)**2 + (pb_start_y - target_y)**2)**0.5
        if dist < 10:
            current_node += 1
            print(f"Reach node {current_node}, moving to next...")

        # Plot (if needed)
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        print(f"FPS: {fps:.2f}")

        cv2.imshow("ONNX YOLOv5 - PiCam", frame)
        plot_time = time.time()
        print(f"plot: {plot_time-cal_time}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    picam2.stop()
    cv2.destroyAllWindows()
    platform.cleanup()
