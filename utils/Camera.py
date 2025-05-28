from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640,480)})
picam2.configure(config)
picam2.start()

prev_time = time.time()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Frame", frame)
    
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    print(f"FPS: {fps:.2f}")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()