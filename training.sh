# python ./yolov5/train.py --epochs 200 --data ./custom_dataset_PB_gray/custom_dataset_PB_gray.yaml --project ./custom_model/PB

python ./yolov5/train.py --cfg yolov5n.yaml --weights yolov5n.pt --data ./custom_dataset_PB/custom_dataset_PB.yaml --epochs 200 --project ./custom_model/augmented

