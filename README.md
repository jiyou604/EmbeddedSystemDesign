# YOLO-Based Particle-Armored Liquid Robot Control Embedded System
This project implements an embedded system that controls the movement of a **particle-armored liquid robot (PB)** by using **YOLOv5 for object detection** and **stepper motors with PID control** to tilt the platform.

---

## 📁 Project Structure

- `custom_dataset_PB/` – Custom YOLO dataset for PB detection  
- `custom_model/` – YOLOv5 training configuration and models  
- `onnxruntime/` – ONNX model export and inference  
- `input/` – Input images and videos  
- `output/` – Inference results  
- `control/` – Stepper motor control and PID logic  
- `utils/` – Helper functions  
- `yolov5/` – Original YOLOv5 source code  

---

## 1. Training

1. **Installation Steps**

Download the source code of YOLOv5 and install the required dependencies according to the instructions in its `README.md` file:

```bash
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

2. **Data Preparation**

Use [labelImg](https://github.com/tzutalin/labelImg) for data labeling.

```bash
pip install labelImg
labelImg
# After labeling, move the labeled images and their corresponding `.txt` files into the appropriate `train`, `valid`, and `test` folders.
```

**Note:** Make sure to label in **YOLO** format, not XML.

3. **Model Training**

Run `training.sh` to start training the model.
When training specific model, please make sure you change "nc:" in .yaml file.

```bash
sh training.sh
```

4. **Export**  

Export trained models to ONNX format for lightweight inference(higher FPS):

```bash
sh utils/export.sh
```

## 2. Detection

Run inference on video or image input to detect the PB in real time using YOLOv5 or ONNX.

**Options and Performance**

1. Standard YOLOv5 Inference
Run using the ObjectDetection.py script inside the utils/ directory.

```bash
python utils/ObjectDetection.py
```

- Lower FPS (~0.5 FPS)

2. ONNX Inference
Lightweight model inference using ONNXRuntime.

```bash
python utils/onnx_infer.py
```

- Higher performance (~2 FPS)
- Requires ONNX model (export.sh creates it)

3. ONNX + Quantization (FP16)
For even faster inference with minimal accuracy drop.

```bash
python utils/onnx_quant.py
```
- Optimized for speed (~4-5 FPS)

4. ONNX + Multithreaded Inference
Implements pipeline-based multithreading for overlapping pre-processing, and post-processing on the inference.

```bash
python onnxruntime/onnx_multithread_infer.py
```
- Fastest (~8 FPS)
- Due to asynchronous threading, inference does not run on every frame, which can lead to intermittent updates.

## 3. Control

Move the PB across a platform by tilting it via 4 stepper motors, based on PID-controlled feedback from object coordinates.

**Hardware**
- Stepper Motor: 28BYJ-48
- Driver Module: ULN2003
- Control Board: Raspberry Pi 4

### Example
```python
import Stepper

pins = [17, 18, 27, 22]

motor1 = Stepper.motor(pins)
motor1.rotate(<traget_step>)# Rotate motor to adjust platform slope
```
