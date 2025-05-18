import cv2
import numpy as np
import os

input_dir = "./images"
output_dir = "./sobel_outputs"
# laplacian_dir = "./laplacian_outputs"
os.makedirs(output_dir, exist_ok=True)
# os.makedirs(laplacian_dir, exist_ok=True)

'''
if os.path.exists(output_dir):
    pass
'''
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

        sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))

        # blurred = cv2.GaussianBlur(sobel, (3, 3), 0)
        # laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        base, ext = os.path.splitext(fname)
        save_path = os.path.join(output_dir, f"sobel_{base}_rot{angle}{ext}")
        cv2.imwrite(save_path, sobel)
        print(f"save: {save_path}")

        # laplacian_abs = np.absolute(laplacian)
        # max_val = np.max(laplacian_abs)
        # if max_val == 0:
        #     laplacian_8u = np.zeros_like(laplacian_abs, dtype=np.uint8)
        # else:
        #     laplacian_8u = np.uint8(255 * laplacian_abs / max_val)

        # lap_name = f"laplacian_{base}_rot{angle}{ext}"
        # lap_path = os.path.join(laplacian_dir, lap_name)
        # cv2.imwrite(lap_path, laplacian_8u)
        # print(f"save: {lap_path}")


