import cv2
import ultralytics
from ultralytics import YOLO
import torch

model_path ='./custom_model/PB/exp2/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.5
model.eval()

file = './input/PB_moving.mp4'
video = cv2.VideoCapture(file)

w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = video.get(cv2.CAP_PROP_FPS)
output_path = './output/output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))


while True:
    ret, frame = video.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results.render()[0]
    out.write(annotated_frame)

    cv2.imshow("YOLOv5 Detection", annotated_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()