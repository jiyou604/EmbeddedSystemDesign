from picamera2 import Picamera2
import cv2
import numpy as np
import time

# init Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


while True:
    frame = picam2.capture_array()

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    lower_rgb = np.array([0, 10, 100])   # B, G, R
    upper_rgb = np.array([80, 150, 255])


    mask = cv2.inRange(blurred, lower_rgb, upper_rgb)

    frame = blurred

    # 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # 너무 작은 물체는 무시
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * radius * radius
            circularity = area / circle_area
            if circularity > 0.5:
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (center[0]-30, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Ball Detection", frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cv2.destroyAllWindows()
picam2.stop()
