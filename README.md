# XAI U-Net: Explainable AI for Lung Nodule Segmentation

This repository contains the source code for an Attention U-Net model designed to segment lung nodules from CT scans using the LIDC-IDRI dataset. A core focus of this project is the integration of Explainable AI (XAI) using the **Integrated Gradients (IG)** method to validate the model's predictions and ensure clinical reliability.

## Overview

The primary goal of this repository is not only to achieve high-performance segmentation but also to make the "black-box" nature of deep learning transparent. By implementing Integrated Gradients, we generate high-resolution, pixel-level attribution maps that highlight which features the model relies on to make its predictions. This helps uncover instances of "shortcut learning" or reasoning anomalies where a model might be correct for the wrong reasons.

## Key Features

- **Attention U-Net**: An advanced U-Net architecture incorporating attention gates to focus on relevant spatial features while suppressing irrelevant background noise.
- **Integrated Gradients (IG)**: A robust XAI technique that satisfies key axioms like sensitivity and implementation invariance. It attributes the prediction to its input features by integrating gradients along a straight line path from a baseline (zero image) to the input.
- **Custom Loss Functions**: Utilizes Focal Tversky Loss to effectively handle extreme class imbalances typical in small lung nodule segmentation.
- **Data Augmentation**: Incorporates techniques like Elastic Deformation to simulate natural anatomical variations.

## Scripts Description

- `lidc_nodule_detection.py`: Main pipeline for loading DICOM data, parsing XML annotations, data augmentation, and training the Attention U-Net.
- `integrated_gradients_unet.py`: The core implementation of the Integrated Gradients method tailored for U-Net segmentation outputs.
- `standalone_ig_example.py`: A standalone executable script to run XAI analysis (single sample, batch, or full dataset) and generate overlay visualizations.
- `attention_unet_test.py` & `simple_attention_test.py`: Scripts for testing model performance and extracting/visualizing internal attention maps.
- `comprehensive_test_metrics.py`: Evaluates the model comprehensively using Dice Coefficient, Intersection over Union (IoU), Precision, and Recall.
- `diagnose_model_performance.py` & `final_model_test.py`: Diagnostic and final evaluation scripts to tune the segmentation threshold.
- `batch_with_visualizations.py`: Utility to process batches of test data with XAI overlay generation.
- `check_reproducibility.py`: Ensures model output stability across different runs.
- `inspect_model_layers.py`: Inspects layer names and structures in the `.h5` model file.

## Dependencies

- TensorFlow / Keras
- NumPy, Pandas, SciPy
- OpenCV, scikit-image
- Matplotlib, Seaborn
- pydicom
- albumentations
- tf-keras-vis
