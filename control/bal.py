from PID import PID
import Stepper

from picamera2 import Picamera2
import RPi.GPIO as GPIO
import cv2
import time
import numpy as np

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


picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# parameters
center_x = 320
center_y = 320
Kp, Ki, Kd = [0.09, 0.0, 0.11]
# Kp, Ki, Kd = [0.09, 0.0, 0.11]
# Kp, Ki, Kd = [0.033, 0.0, 0.08]
max_rotation = 1000
toggle_threshold = 10

# PID
pid_x = PID(kp=Kp, ki=Ki, kd=Kd)
pid_y = PID(kp=Kp, ki=Ki, kd=Kd)

# motor classes
GPIO.setmode(GPIO.BCM)
motor_x0 = Stepper.Motor([14, 15, 17, 18], max_step=max_rotation)
motor_x1 = Stepper.Motor([27, 22, 23, 24], max_step=max_rotation)
motor_y0 = Stepper.Motor([10, 9, 25, 11], max_step=max_rotation)
motor_y1 = Stepper.Motor([16, 26, 20, 21], max_step=max_rotation)

platform = Stepper.Platform([motor_x0, motor_x1, motor_y0, motor_y1])

last_steps = [0, 0]

try:
    while True:
        frame = picam2.capture_array()
        output = get_position(frame)

        if output is not None:
            center, radius = output
            error = [center_x - center[0], center_y - center[1]]

            print(center)
            x_output = pid_x.compute(center_x - center[0])
            y_output = pid_y.compute(center_y - center[1])       

            x_steps = int(-x_output)
            y_steps = int(y_output)

            # platform.tilt(x_steps, y_steps)

            # toggle mode
            if error[0] < toggle_threshold:
                toggle_mode_x = True
            else:
                toggle_mode_x = False
            if error[1] < toggle_threshold:
                toggle_mode_y = True
            else:
                toggle_mode_y = False

            platform.tilt(x_steps, y_steps, toggle_mode_x=toggle_mode_x, toggle_mode_y=toggle_mode_y)

            # debuging
            print("PID_x: ", pid_x.last_error, pid_x.integral, pid_x.derivative, " | ", "PID_y: ", pid_y.last_error, pid_y.integral, pid_y.derivative)
            print("Steps: ", x_steps, y_steps)

            # time.sleep(0.005) 

        else:
            print("Ball not detected.")
            steps_x = int(last_steps[0] * -0.5)
            steps_y = int(last_steps[1] * -0.5)


except KeyboardInterrupt:
    picam2.stop()
    platform.cleanup()


