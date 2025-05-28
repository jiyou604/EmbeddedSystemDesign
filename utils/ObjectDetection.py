import cv2
import torch
import time
from picamera2 import Picamera2

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

print("picam working")

prev_time = time.time()

model_path ='./EmbeddedSystemDesign/custom_model/PB/exp2/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()


print("yolo upload done")

while True:
    print("true")
    frame = picam2.capture_array()
    results = model(frame)
    annotated_frame = results.render()[0]

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    print(f"FPS: {fps:.2f}")

    cv2.imshow("YOLOv5n Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()