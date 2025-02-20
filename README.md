## Overview
This repository contains an implementation of the YOLO (You Only Look Once) object detection system, based on the research paper "YOLO: Unified, Real-Time Object Detection" by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi ([link to the paper](https://arxiv.org/abs/1506.026403)). YOLO is a state-of-the-art, real-time object detection system that reframes object detection as a single regression problem, directly predicting bounding boxes and class probabilities from full images in one evaluation. Its unified architecture allows for fast and accurate detection, suitable for applications such as autonomous driving, robotics, and assistive devices.

## Features
- **Real-Time Performance:** YOLO processes images at 45 frames per second with the base model and 155 frames per second with the Fast YOLO variant.
- **Unified Architecture:** A single convolutional neural network predicts multiple bounding boxes and class probabilities.
- **Generalization:** YOLO outperforms traditional methods when generalizing from natural images to other domains like artwork.

## Requirements
To run the code, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- Scipy
- Pytorch

## Usage
for training:
```
from yolo import YOLO
from loss import YOLO_Loss

model = YOLO()
yolo_loss = YOLO_Loss(5, 0.5)
# get images with shape(batch, 3, 448, 448)
# get main boxes with shape (batch, n, 4)
# get grid_probs_main with shape (batch, n, 7, 7, 20)
boxes, boxes_confidence, grid_probs = model(images)
for i in range(batch):
    loss = yolo_loss(main_boxes[i] boxes[i], boxes_confidence[i], grid_probs_main[i], grid_probs[i])
    loss.backward()
```
for inference:
```
from yolo import YOLO
from utils import postprocess

model = YOLO()
# get images with shape(batch, 3, 448, 448)
boxes, boxes_confidence, grid_probs = model(images)
for i in range(batch):
    final_boxes = postprocess(boxes[i], boxes_confidence[i], grid_probs[i], 0.7)
```

  
