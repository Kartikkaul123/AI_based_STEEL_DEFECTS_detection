## AI_based_STEEL_DEFECTS_detection
AI-based steel surface defect detection system using YOLOv8. The model detects and localizes multiple steel surface defects in real-time for automated industrial quality inspection.

#Overview

This project implements an AI-based steel surface defect detection system using the YOLOv8 object detection framework. The model is trained on multiple steel defect datasets to detect and localize defects such as scratches, inclusions, pitted surfaces, stains, and rolling defects. The system is designed for automated quality inspection in steel manufacturing environments, particularly for conveyor-based inspection systems.

#Features

i)Real-time steel surface defect detection

ii)Multi-class defect classification

iii)Bounding box localization of defects

iv)Dataset merging for improved robustness

v)Suitable for industrial inspection pipelines

#Datasets

The model was trained using a combination of publicly available steel defect datasets to improve diversity and generalization.

NEU Steel Surface Defect Dataset

Contains six common steel surface defects including:

1)Crazing

2)Inclusion

3)Patches

4)Pitted Surface

5)Scratches

6)Rolled-in Scale

GC10 Steel Defect Dataset

A high-resolution dataset containing multiple defect categories with more complex textures and real industrial conditions.

Severstal Steel Defect Dataset
A real-world dataset from Kaggle originally designed for segmentation, which was adapted for object detection by converting annotations to bounding boxes.
1)Crease 
2)Oil Spot
3)Water Spot
4)Silk Spot 
5)Waist Folding
6)Rolled Pit

The datasets were merged to create a larger and more diverse training dataset.

#Model

The project uses YOLOv8-M (Medium) from the Ultralytics YOLO framework.

Why YOLOv8-M

i)Balanced accuracy and speed

ii)Suitable for real-time industrial inspection

iii)Efficient GPU utilization

Training Configuration

Epochs: 80

Image Size: 704

Batch Size: 16

Augmentation: Mosaic, scaling, flipping, HSV variation

Performance
mAP@50: ~0.82
mAP@50-95: ~0.64

# Trained Model
Download the trained YOLOv8 model:
https://github.com/Kartikkaul123/AI_based_STEEL_DEFECTS_detection/releases/tag/v1.0
