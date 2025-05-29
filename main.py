import cv2
import os
import ultralytics
from ultralytics import YOLO
import torch

model_path ='./custom_model/PB/exp2/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
# 저신뢰 bounding box 아예 제거
model.conf = 0.8
model.eval()

input_dir = './input'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith('.mp4'):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        print(f"Skipping {fname}, output already exists.")
        continue

    cap = cv2.VideoCapture(in_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_path = os.path.join(output_dir, fname)
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        #영상도 gray로 읽기
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # YOLO 추론 및 렌더링
        results = model(gray)
        # 서보모터 제어용 conf 및 box 위치 받기
        boxes = results.xyxy[0]
        for *xyxy, conf, cls in boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2)/2
            cy = (y1 + y2)/2
            # 확인용
            print(f"conf={conf:.2f}, center box=({cx},{cy})")

        rendered = results.render()[0]
        out.write(rendered)

    cap.release()
    out.release()
    print(f"Processed and saved: {out_path}")

cv2.destroyAllWindows()

'''
#서보모터 제어로직
import pigpio
import time

# pigpio 초기화
pi = pigpio.pi()
PAN_PIN = 17    # 가로(팬) 서보 연결 핀
TILT_PIN = 18   # 세로(틸트) 서보 연결 핀

def angle_to_pulse(angle):
    return 500 + int((angle / 180.0) * 2000)

# 값 범위 매핑
def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# 화면 크기
frame_w, frame_h = w, h  # 전역으로 가져왔다고 가정

tilt_max = 30  # 최대 리프트 각도 차이
tilt_x = min(abs(dx)/center_x * tilt_max, tilt_max)
tilt_y = min(abs(dy)/center_y * tilt_max, tilt_max)

# 판 들어올리는 lifting
def lift(servo, angle):
    pi.set_servo_pulsewidth(pin[servo], angle_to_pulse(90 + angle))
def reset(servo):
    pi.set_servo_pulsewidth(pin[servo], angle_to_pulse(90))

# 초기화
for s in ['east','west','north','south']: reset(s)

if dx < 0: lift('east', tilt_x)
elif dx > 0: lift('west', tilt_x)

if dy < 0: lift('south', tilt_y)
elif dy > 0: lift('north', tilt_y)

vibe_times = 0.02

if dx != 0 and dy != 0:
    for _ in range(vibe_times):
        lift('east', tilt_x)
        lift('south', tilt_y)
        time.sleep(vibe_delay)
        reset('east')
        reset('south')
        time.sleep(vibe_delay)


time.sleep(0.02)
'''
