# YOLO-Based Particle-Armored Liquid Robot Control Embedded System

## Installation Steps

Download the source code of YOLOv5 and install the required dependencies according to the instructions in its `README.md` file:

```bash
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
````

## Data Preparation

Use [labelImg](https://github.com/tzutalin/labelImg) for data labeling.

```bash
pip install labelImg
labelImg
# After labeling, move the labeled images and their corresponding `.txt` files into the appropriate `train`, `valid`, and `test` folders.
```

**Note:** Make sure to label in **YOLO** format, not XML.

## Training

Run `training.sh` to start training the model.

```bash
sh training.sh
```

To test the model on an image and check the results, run `inference.sh`.

```bash
sh inference.sh
```

## Run

To run YOLO detection on a video and save the results:

1. Place your input video in the `input` directory.

2. Run the main program:

```bash
python main.py
```
