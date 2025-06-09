import torch
import cv2
import numpy as np
from AStar.py import a_star

model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model/Obstacles/exp2/weights/best.pt')

img = cv2.imread('background.jpg') # at start, PiCam picutres the map for seaching the path
# now, not exist background.jpg
# for debugging, check

if img is None:
    raise FileNotFoundError('background.jpg not found')

results = model(img)
boxes = results.xyxy[0].cpu().numpy()  # [[x1, y1, x2, y2, conf, cls], ...]
# bbox export

# transform binary map, black :0, free :1, so we view the black dots as obstacles
H, W = img.shape[:2]
grid = np.ones((H, W), dtype=np.uint8) # transporm jpg to grid map for astar approach

# in bbox region, '0' cannot be used
for *xyxy, conf, cls in boxes:
    x1, y1, x2, y2 = map(int, xyxy)
    grid[y1:y2, x1:x2] = 0  # obstacle

start = (H-1, 0)
goal  = (0, W-1)

path = a_star(grid, start, goal)

out_img = img.copy()

for y, x in path:
    cv2.circle(out_img, (x, y), 1, (255, 0, 0), -1)  
cv2.circle(out_img, (start[1], start[0]), 4, (0,255,0), -1)  
cv2.circle(out_img, (goal[1],  goal[0]), 4, (0,0,255), -1) 
cv2.imwrite('astar_yolo_path.png', out_img)

# for debugging, check the binary map
cv2.imwrite('binary_map.png', grid * 255)
# save the path.txt
with open("astar_path.txt", "w") as f:
    for y, x in path:
        f.write(f"{y},{x}\n")
# save as numpy list
np.savez("astar_path.npz", path=np.array(path))


