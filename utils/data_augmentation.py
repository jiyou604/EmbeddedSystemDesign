import os
import cv2
import numpy as np
import random
from glob import glob

# 사용자 정의 경로
BASE_DIR = 'custom_dataset_PB/train'
INPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'images')
INPUT_LABEL_DIR = os.path.join(BASE_DIR, 'labels')
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, '../augmented/images')
OUTPUT_LABEL_DIR = os.path.join(BASE_DIR, '../augmented/labels')

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def yolo_to_bbox(yolo_label, img_shape):
    h, w = img_shape
    cls, x, y, bw, bh = map(float, yolo_label.strip().split())
    x1 = int((x - bw / 2) * w)
    y1 = int((y - bh / 2) * h)
    x2 = int((x + bw / 2) * w)
    y2 = int((y + bh / 2) * h)
    return int(cls), x1, y1, x2, y2

def bbox_to_yolo(cls, x1, y1, x2, y2, img_shape):
    h, w = img_shape
    x = ((x1 + x2) / 2) / w
    y = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"

def random_crop(image, bboxes, fname_base, crop_scale=0.8):
    h, w = image.shape[:2]
    new_w, new_h = int(w * crop_scale), int(h * crop_scale)
    x_start = random.randint(0, w - new_w)
    y_start = random.randint(0, h - new_h)
    x_end = x_start + new_w
    y_end = y_start + new_h

    cropped = image[y_start:y_end, x_start:x_end]
    cropped_bboxes = []

    for label in bboxes:
        cls, x1, y1, x2, y2 = yolo_to_bbox(label, (h, w))
        new_x1 = max(x1, x_start)
        new_y1 = max(y1, y_start)
        new_x2 = min(x2, x_end)
        new_y2 = min(y2, y_end)

        if new_x1 >= new_x2 or new_y1 >= new_y2:
            continue

        adj_x1 = new_x1 - x_start
        adj_y1 = new_y1 - y_start
        adj_x2 = new_x2 - x_start
        adj_y2 = new_y2 - y_start

        cropped_bboxes.append(bbox_to_yolo(cls, adj_x1, adj_y1, adj_x2, adj_y2, (new_h, new_w)))

    if cropped_bboxes:
        cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{fname_base}_crop.jpg", cropped)
        with open(f"{OUTPUT_LABEL_DIR}/{fname_base}_crop.txt", 'w') as f:
            f.write("\n".join(cropped_bboxes))

def apply_augmentations(image, bboxes, fname_base):
    h, w = image.shape[:2]

    # Horizontal Flip
    flipped = cv2.flip(image, 1)
    flipped_bboxes = []
    for label in bboxes:
        cls, x1, y1, x2, y2 = yolo_to_bbox(label, (h, w))
        new_x1 = w - x2
        new_x2 = w - x1
        flipped_bboxes.append(bbox_to_yolo(cls, new_x1, y1, new_x2, y2, (h, w)))
    cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{fname_base}_flip.jpg", flipped)
    with open(f"{OUTPUT_LABEL_DIR}/{fname_base}_flip.txt", 'w') as f:
        f.write("\n".join(flipped_bboxes))

    # Rotations
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        rotated_bboxes = []
        for label in bboxes:
            cls, x1, y1, x2, y2 = yolo_to_bbox(label, (h, w))
            points = np.array([[x1, y1], [x2, y2]])
            ones = np.ones((2, 1))
            points = np.hstack([points, ones])
            rotated_points = points @ M.T
            rx1, ry1 = np.min(rotated_points, axis=0)
            rx2, ry2 = np.max(rotated_points, axis=0)
            rotated_bboxes.append(bbox_to_yolo(cls, int(rx1), int(ry1), int(rx2), int(ry2), (h, w)))
        cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{fname_base}_rot{angle}.jpg", rotated)
        with open(f"{OUTPUT_LABEL_DIR}/{fname_base}_rot{angle}.txt", 'w') as f:
            f.write("\n".join(rotated_bboxes))

    # Brightness
    for factor in [0.5, 1.5]:
        bright = adjust_brightness(image, factor)
        cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{fname_base}_bright{int(factor*100)}.jpg", bright)
        with open(f"{OUTPUT_LABEL_DIR}/{fname_base}_bright{int(factor*100)}.txt", 'w') as f:
            f.write("\n".join(bboxes))

    # Random Crop
    random_crop(image, bboxes, fname_base)

def augment_dataset():
    image_paths = glob(os.path.join(INPUT_IMAGE_DIR, "*.jpg"))
    print(f"Found {len(image_paths)} images to augment.")
    for img_path in image_paths:
        base_name = os.path.basename(img_path).replace('.jpg', '')
        label_path = os.path.join(INPUT_LABEL_DIR, base_name + '.txt')
        if not os.path.exists(label_path):
            print(f"Label not found for {base_name}, skipping.")
            continue

        image = cv2.imread(img_path)
        with open(label_path, 'r') as f:
            bboxes = f.read().strip().splitlines()

        apply_augmentations(image, bboxes, base_name)

if __name__ == "__main__":
    augment_dataset()
