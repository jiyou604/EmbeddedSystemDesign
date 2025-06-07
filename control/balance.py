import cv2
import Stepper
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2


def detect_ball(frame, min_area=55):
    lower_blue = np.array([50, 50, 50])   # 진한 파랑
    upper_blue = np.array([255, 255, 255]) # 연한 파랑

    img_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    mask = cv2.inRange(img_blur, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        circularity = area / circle_area
        if circularity > 0.65:
            centers.append((int(x), int(y)))

        # print(centers)
    return centers

GPIO.setmode(GPIO.BCM)
pins_x0 = [14, 15, 17, 18]
pins_x1 = [27, 22, 23, 24]
pins_y0 = [10, 9, 25, 11]
pins_y1 = [16, 26, 20, 21]

motor_x0 = Stepper.Motor(pins_x0)
motor_x1 = Stepper.Motor(pins_x1)
motor_y0 = Stepper.Motor(pins_y0)
motor_y1 = Stepper.Motor(pins_y1)

x_axis = Stepper.MotorPair([motor_x0, motor_x1])
y_axis = Stepper.MotorPair([motor_y0, motor_y1])

threshold = 0.5

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()

    centers = detect_ball(frame)

    for (x, y) in centers:
        cv2.circle(frame, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(frame, f"{x},{y}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("Connectors", frame)

    # [(60, 413), (336, 101)]



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
