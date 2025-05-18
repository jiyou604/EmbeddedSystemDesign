import cv2
import os
import numpy as np

input_dir = "./images"
output_dir = "./BSL_outputs"
os.makedirs(output_dir, exist_ok=True)

angles = [0, 90, 180, 270]

for fname in os.listdir(input_dir):
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    for angle in angles:
        if angle == 0:
            rotated = img.copy()
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = rotated.shape
        resized = cv2.resize(rotated, (int(w * 0.3), int(h * 0.3)))

        blurred = cv2.GaussianBlur(resized, (3, 3), 0)

        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))

        laplacian = cv2.Laplacian(sobel, cv2.CV_64F)

        base, ext = os.path.splitext(fname)
        save_path = os.path.join(output_dir, f"final_{base}_rot{angle}{ext}")
        cv2.imwrite(save_path, laplacian)
        print(f"save: {save_path}")
