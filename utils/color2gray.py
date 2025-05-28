import os
import cv2

# 입력 폴더와 출력 폴더 경로
input_folder = './EmbeddedSystemDesign/custom_dataset_PB/train/images'
output_folder = 'gray_PB'

# 출력 폴더가 없다면 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더의 모든 파일 반복
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 이미지 불러오기
        img = cv2.imread(input_path)

        if img is None:
            print(f"이미지를 불러올 수 없음: {filename}")
            continue

        # 흑백으로 변환
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 저장
        cv2.imwrite(output_path, gray_img)
        print(f"변환 완료: {filename}")
