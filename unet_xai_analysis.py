"""
U-Net XAI (Explainable AI) Analysis System """

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Machine Learning imports
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Medical imaging imports
import skimage
from skimage.draw import polygon
import pydicom
import cv2
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from skimage.morphology import remove_small_objects

# XML and file handling
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
from typing import List, Tuple, Optional
import random
from collections import defaultdict
import gc
import joblib
import pickle

# TensorFlow/Keras imports
import tensorflow as tf

# SHAP and visualization
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print
import albumentations as A

print("All imports completed successfully!")


class Config:
    """Configuration class for the U-Net XAI analysis system"""
    DATA_DIR = "/workspace/storage/lidc_dataset"
    METADATA_PATH = "workspace/storage/LIDC-IDRI_MetaData.csv"
    OUTPUT_DIR = "workspace/output_latest"
    
    # Image processing parameters
    IMAGE_SIZE = (512, 512)
    PATCH_SIZE = 32
    STRIDE = 32
    HU_MIN = -1000
    HU_MAX = 400
    
    # Training parameters
    BATCH_SIZE = 8
    PATCH_BATCH_SIZE = 16
    EPOCHS = 1
    LEARNING_RATE = 5e-4
    VALIDATION_SPLIT = 0.2
    
    # Model parameters
    UNET_FILTERS = [32, 64, 128, 256]
    DROPOUT_RATE = 0.5
    
    # Data balance parameters
    NUM_PATIENTS = 10
    MAX_SLICES_PER_PATIENT = 30
    NODULE_CONTEXT_SLICES = 10
    
    # Augmentation parameters
    AUGMENTATION_PROBABILITY = 0.8
    
    # Thresholds
    SEGMENTATION_THRESHOLD = 0.4
    MALIGNANCY_THRESHOLD = 3  # 1-3: benign, 4-5: malignant
    
    @classmethod
    def create_directories(cls):
        """Create necessary output directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "results"), exist_ok=True)


def patient_root(pid: int) -> str:
    """Get patient directory path based on patient ID"""
    if 1 <= pid <= 200: 
        rng = "LIDC-IDRI-0001-0200"
    elif 201 <= pid <= 400: 
        rng = "LIDC-IDRI-0201-0400"
    elif 401 <= pid <= 600: 
        rng = "LIDC-IDRI-0401-0600"
    else: 
        raise ValueError("Patient ID must be 1-600")
    return os.path.join(Config.DATA_DIR, rng, f"LIDC-IDRI-{pid:04d}")


def normalize_hu(image, hu_min=-1000, hu_max=400):
    """Normalize HU values to 0-1 range"""
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image.astype(np.float32)


def load_dicom_series_enhanced(dicom_files):
    """Enhanced DICOM loading with better error handling"""
    if not dicom_files:
        return []
    
    loaded_data = []
    for filepath in tqdm(dicom_files, desc="Loading DICOM files"):
        try:
            dcm = pydicom.dcmread(filepath)
            
            # Extract image data
            img = dcm.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept if available
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                img = img * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Get slice position for sorting
            slice_location = getattr(dcm, 'SliceLocation', 0)
            instance_number = getattr(dcm, 'InstanceNumber', 0)
            
            uid = dcm.SOPInstanceUID
            
            loaded_data.append({
                'image': img,
                'uid': uid,
                'filepath': filepath,
                'slice_location': slice_location,
                'instance_number': instance_number
            })
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    # Sort by slice location or instance number
    loaded_data.sort(key=lambda x: (x['slice_location'], x['instance_number']))
    
    return loaded_data


def parse_lidc_xml_enhanced(xml_path):
    """Enhanced XML parsing with better error handling"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {'idri': root.tag.split('}')[0].strip('{')}
        
        nodule_data = {}
        
        # Parse reading sessions
        for session in root.findall('.//idri:readingSession', ns):
            for nodule in session.findall('.//idri:unblindedReadNodule', ns):
                
                # Extract nodule characteristics
                characteristics = {}
                char_elem = nodule.find('.//idri:characteristics', ns)
                if char_elem is not None:
                    for char in ['malignancy', 'subtlety', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture']:
                        elem = char_elem.find(f'idri:{char}', ns)
                        if elem is not None and elem.text:
                            try:
                                characteristics[char] = int(elem.text.strip())
                            except ValueError:
                                characteristics[char] = 0
                
                # Extract ROI information
                for roi in nodule.findall('.//idri:roi', ns):
                    sop_elem = roi.find('idri:imageSOP_UID', ns)
                    if sop_elem is None or sop_elem.text is None:
                        continue
                    
                    uid = sop_elem.text.strip()
                    
                    # Extract polygon coordinates
                    polygon_coords = []
                    for edge in roi.findall('.//idri:edgeMap', ns):
                        y_elem = edge.find('idri:yCoord', ns)
                        x_elem = edge.find('idri:xCoord', ns)
                        
                        if y_elem is not None and x_elem is not None:
                            try:
                                y = int(y_elem.text.strip())
                                x = int(x_elem.text.strip())
                                polygon_coords.append((y, x))
                            except (ValueError, TypeError):
                                continue
                    
                    if polygon_coords:
                        if uid not in nodule_data:
                            nodule_data[uid] = []
                        
                        nodule_data[uid].append({
                            'polygon': polygon_coords,
                            'characteristics': characteristics
                        })
        
        return nodule_data
        
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return {}


def create_mask_from_nodule_data(nodule_data, image_shape):
    """Create binary mask and extract characteristics"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    malignancy_scores = []
    
    for nodule_info in nodule_data:
        polygon_coords = nodule_info['polygon']
        characteristics = nodule_info['characteristics']
        
        if len(polygon_coords) >= 3:
            # Create polygon mask
            rr, cc = polygon(
                [p[0] for p in polygon_coords], 
                [p[1] for p in polygon_coords], 
                image_shape
            )
            mask[rr, cc] = 1
            
            # Extract malignancy score
            malignancy = characteristics.get('malignancy', 0)
            malignancy_scores.append(malignancy)
    
    # Return the highest malignancy score if multiple nodules
    final_malignancy = max(malignancy_scores) if malignancy_scores else 0
    
    return mask, final_malignancy


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Compute Dice coefficient"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for handling class imbalance"""
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        focal_loss = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fixed


def iou(y_true, y_pred, smooth=1e-6):
    """Compute Intersection over Union (IoU) metric"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    return (intersection + smooth) / (union + smooth)


def focal_tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, gamma=0.75, smooth=1e-6):
    """Focal Tversky loss variant robust to class imbalance"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    return tf.pow((1 - tversky_index), gamma)


def load_saved_data():
    """Load saved data files for XAI analysis"""
    print("Loading saved data files...")
    
    # Load comprehensive results
    comprehensive_path = os.path.join(Config.OUTPUT_DIR, 'comprehensive_results.npz')
    comprehensive_data = np.load(comprehensive_path)
    print("Loaded comprehensive_results.npz")
    
    # Load detailed metadata
    metadata_path = os.path.join(Config.OUTPUT_DIR, 'detailed_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        detailed_metadata = pickle.load(f)
    print("Loaded detailed_metadata.pkl")
    
    # Load training data with metadata
    training_path = os.path.join(Config.OUTPUT_DIR, 'training_data_with_metadata.npz')
    training_data = np.load(training_path)
    print("Loaded training_data_with_metadata.npz")
    
    print(f"Data shapes:")
    print(f"  X_test: {comprehensive_data['X_test'].shape}")
    print(f"  y_test (ground truth): {comprehensive_data['y_test'].shape}")
    print(f"  pred_masks (U-Net predictions): {comprehensive_data['pred_masks'].shape}")
    
    return comprehensive_data, detailed_metadata, training_data


def get_sample_info(comprehensive_data, detailed_metadata, image_index):
    """Extract sample information for XAI analysis"""
    test_metadata = detailed_metadata['test_metadata']
    sample_metadata = test_metadata[image_index]
    
    # Get classification label
    class_label = comprehensive_data['y_test'][image_index]
    
    # Get malignancy score from metadata
    malignancy_score = sample_metadata.get('malignancy', 0)
    
    return sample_metadata, class_label, malignancy_score


def get_actual_ground_truth_for_xai(sample_metadata):
    """Retrieve actual ground truth mask for XAI analysis"""
    try:
        # Get patient directory
        patient_id = sample_metadata['patient_id']
        patient_dir = patient_root(patient_id)
        
        # Load DICOM files
        dicom_files = [os.path.join(r, f) for r, _, files in os.walk(patient_dir) 
                      for f in files if f.endswith('.dcm')]
        
        if not dicom_files:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
        
        dicom_data = load_dicom_series_enhanced(dicom_files)
        if not dicom_data:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
        
        # Find the specific slice
        uid_to_idx = {data['uid']: idx for idx, data in enumerate(dicom_data)}
        target_uid = sample_metadata['uid']
        
        if target_uid not in uid_to_idx:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
        
        slice_idx = uid_to_idx[target_uid]
        original_image = dicom_data[slice_idx]['image']
        
        # Load XML files and parse nodule data
        xml_files = [os.path.join(r, f) for r, _, files in os.walk(patient_dir) 
                    for f in files if f.endswith('.xml')]
        
        nodule_data = {}
        for xml_file in xml_files:
            nodule_data.update(parse_lidc_xml_enhanced(xml_file))
        
        # Create mask if nodule data exists for this UID
        if target_uid in nodule_data:
            mask, _ = create_mask_from_nodule_data(nodule_data[target_uid], original_image.shape)
            mask_resized = cv2.resize(mask, Config.IMAGE_SIZE)
            return mask_resized.astype(np.uint8)
        else:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
            
    except Exception as e:
        print(f"Error retrieving ground truth: {e}")
        return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)


def improved_gradient_explanation(unet_model, image_to_explain, target_mode='max_confidence'):
    """
    Improved gradient-based explanation for U-Net segmentation.
    
    Args:
        unet_model: Trained U-Net model
        image_to_explain: Input image (batch of 1)
        target_mode: Target strategy ('max_confidence', 'total_area', 'mean_confidence')
    
    Returns:
        gradients: Computed gradients
        prediction: Model prediction
    """
    # Ensure input is a tensor
    if not tf.is_tensor(image_to_explain):
        image_to_explain = tf.convert_to_tensor(image_to_explain, dtype=tf.float32)
    
    # Track gradients
    with tf.GradientTape() as tape:
        tape.watch(image_to_explain)
        
        # Get model prediction
        prediction = unet_model(image_to_explain)
        
        # Compute target based on mode
        if target_mode == 'max_confidence':
            target = tf.reduce_max(prediction)
        elif target_mode == 'total_area':
            target = tf.reduce_sum(prediction)
        elif target_mode == 'mean_confidence':
            target = tf.reduce_mean(prediction)
        else:
            target = tf.reduce_max(prediction)
    
    # Compute gradients
    gradients = tape.gradient(target, image_to_explain)
    
    return gradients, prediction


def integrated_gradients_explanation(unet_model, image_to_explain, baseline=None, steps=1000, target_mode='max_confidence'):
    """
    Integrated Gradients explanation for U-Net segmentation.
    
    Args:
        unet_model: Trained U-Net model
        image_to_explain: Input image (batch of 1)
        baseline: Baseline image (default: zeros)
        steps: Number of integration steps
        target_mode: Target strategy
    
    Returns:
        integrated_grads: Computed integrated gradients
        prediction: Model prediction
    """
    # Create baseline if not provided
    if baseline is None:
        baseline = tf.zeros_like(image_to_explain)
    
    # Ensure inputs are tensors
    if not tf.is_tensor(image_to_explain):
        image_to_explain = tf.convert_to_tensor(image_to_explain, dtype=tf.float32)
    if not tf.is_tensor(baseline):
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    
    # Generate interpolated images
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated_images = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (image_to_explain - baseline)
        interpolated_images.append(interpolated)
    
    interpolated_images = tf.concat(interpolated_images, axis=0)
    
    # Compute gradients for all interpolated images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = unet_model(interpolated_images)
        
        # Compute target based on mode
        if target_mode == 'max_confidence':
            targets = tf.reduce_max(predictions, axis=[1, 2, 3])
        elif target_mode == 'total_area':
            targets = tf.reduce_sum(predictions, axis=[1, 2, 3])
        elif target_mode == 'mean_confidence':
            targets = tf.reduce_mean(predictions, axis=[1, 2, 3])
        else:
            targets = tf.reduce_max(predictions, axis=[1, 2, 3])
    
    # Compute gradients
    gradients = tape.gradient(targets, interpolated_images)
    
    # Average gradients (integrate)
    integrated_grads = tf.reduce_mean(gradients, axis=0)
    
    # Multiply by input difference
    integrated_grads = integrated_grads * (image_to_explain - baseline)
    
    # Get original prediction
    prediction = unet_model(image_to_explain)
    
    return integrated_grads, prediction


def compute_segmentation_metrics(prediction, ground_truth, threshold=0.4):
    """Compute segmentation metrics"""
    # Apply threshold to prediction
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = ground_truth.astype(np.uint8)
    
    # Compute metrics
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-6)
    iou = intersection / (union + 1e-6)
    precision = intersection / (np.sum(pred_binary) + 1e-6)
    recall = intersection / (np.sum(gt_binary) + 1e-6)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'gt_area': np.sum(gt_binary),
        'pred_area': np.sum(pred_binary),
        'intersection': intersection
    }


def visualize_comprehensive_analysis(image, ground_truth, unet_prediction, gradients, 
                                   integrated_grads, sample_info, enhancement_factor=1000):
    """
    Create comprehensive visualization of XAI analysis results.
    
    Args:
        image: Original input image
        ground_truth: Ground truth segmentation mask
        unet_prediction: U-Net prediction
        gradients: Gradient-based explanation
        integrated_grads: Integrated gradients explanation
        sample_info: Sample metadata dictionary
        enhancement_factor: Factor to enhance visualization
    """
    # Prepare data
    image_2d = np.squeeze(image)
    gt_2d = np.squeeze(ground_truth)
    pred_2d = np.squeeze(unet_prediction)
    
    # Compute segmentation metrics
    metrics = compute_segmentation_metrics(pred_2d, gt_2d)
    
    # Apply threshold to prediction for visualization
    pred_binary = (pred_2d > Config.SEGMENTATION_THRESHOLD).astype(np.uint8)
    gt_binary = gt_2d.astype(np.uint8)
    
    # Process gradients if available
    if gradients is not None:
        grad_2d = np.squeeze(gradients.numpy())
        grad_enhanced = grad_2d * enhancement_factor
    else:
        grad_2d = np.zeros_like(image_2d)
        grad_enhanced = np.zeros_like(image_2d)
    
    # Process integrated gradients if available
    if integrated_grads is not None:
        ig_2d = np.squeeze(integrated_grads.numpy())
        ig_enhanced = ig_2d * enhancement_factor
    else:
        ig_2d = np.zeros_like(image_2d)
        ig_enhanced = np.zeros_like(image_2d)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    # Row 1: Original images and masks
    axes[0, 0].imshow(image_2d, cmap='gray')
    axes[0, 0].set_title('Original CT Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_2d, cmap='hot')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_2d, cmap='hot')
    axes[0, 2].set_title('U-Net Prediction')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pred_binary, cmap='hot')
    axes[0, 3].set_title('Binary Prediction')
    axes[0, 3].axis('off')
    
    # Overlay predictions
    overlay_pred = np.zeros((*image_2d.shape, 3))
    overlay_pred[:, :, 0] = image_2d  # Red channel for image
    overlay_pred[:, :, 1] = pred_binary  # Green channel for prediction
    axes[0, 4].imshow(overlay_pred)
    axes[0, 4].set_title('Image + Prediction')
    axes[0, 4].axis('off')
    
    # Ground truth overlay
    overlay_gt = np.zeros((*image_2d.shape, 3))
    overlay_gt[:, :, 0] = image_2d  # Red channel for image
    overlay_gt[:, :, 2] = gt_binary  # Blue channel for ground truth
    axes[0, 5].imshow(overlay_gt)
    axes[0, 5].set_title('Image + Ground Truth')
    axes[0, 5].axis('off')
    
    # Row 2: Gradient explanations
    axes[1, 0].imshow(grad_2d, cmap='RdBu_r')
    axes[1, 0].set_title('Raw Gradients')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(grad_enhanced, cmap='RdBu_r')
    axes[1, 1].set_title('Enhanced Gradients')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(ig_2d, cmap='RdBu_r')
    axes[1, 2].set_title('Raw Integrated Gradients')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(ig_enhanced, cmap='RdBu_r')
    axes[1, 3].set_title('Enhanced Integrated Gradients')
    axes[1, 3].axis('off')
    
    # Gradient overlays
    grad_overlay = np.zeros((*image_2d.shape, 3))
    grad_overlay[:, :, 0] = image_2d
    grad_overlay[:, :, 1] = np.abs(grad_enhanced)
    grad_overlay[:, :, 2] = np.abs(grad_enhanced)
    axes[1, 4].imshow(grad_overlay)
    axes[1, 4].set_title('Image + Gradient Overlay')
    axes[1, 4].axis('off')
    
    ig_overlay = np.zeros((*image_2d.shape, 3))
    ig_overlay[:, :, 0] = image_2d
    ig_overlay[:, :, 1] = np.abs(ig_enhanced)
    ig_overlay[:, :, 2] = np.abs(ig_enhanced)
    axes[1, 5].imshow(ig_overlay)
    axes[1, 5].set_title('Image + IG Overlay')
    axes[1, 5].axis('off')
    
    # Row 3: Analysis and statistics
    # Create detailed analysis plots
    axes[2, 0].hist(grad_2d.flatten(), bins=50, alpha=0.7, color='blue')
    axes[2, 0].set_title('Gradient Distribution')
    axes[2, 0].set_xlabel('Gradient Value')
    axes[2, 0].set_ylabel('Frequency')
    
    axes[2, 1].hist(ig_2d.flatten(), bins=50, alpha=0.7, color='green')
    axes[2, 1].set_title('IG Distribution')
    axes[2, 1].set_xlabel('IG Value')
    axes[2, 1].set_ylabel('Frequency')
    
    # Prediction confidence map
    confidence_map = pred_2d
    im = axes[2, 2].imshow(confidence_map, cmap='viridis')
    axes[2, 2].set_title('Prediction Confidence')
    axes[2, 2].axis('off')
    plt.colorbar(im, ax=axes[2, 2])
    
    # Difference between prediction and ground truth
    diff_map = np.abs(pred_binary.astype(float) - gt_binary.astype(float))
    axes[2, 3].imshow(diff_map, cmap='Reds')
    axes[2, 3].set_title('Prediction Error')
    axes[2, 3].axis('off')
    
    # Metrics visualization
    metrics_names = ['Dice', 'IoU', 'Precision', 'Recall']
    metrics_values = [metrics['dice'], metrics['iou'], metrics['precision'], metrics['recall']]
    axes[2, 4].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
    axes[2, 4].set_title('Segmentation Metrics')
    axes[2, 4].set_ylabel('Score')
    axes[2, 4].set_ylim(0, 1)
    
    # Summary statistics
    summary_text = f"""
PATIENT INFO:
Patient ID: {sample_info.get('patient_id', 'N/A')}
Slice Index: {sample_info.get('slice_idx', 'N/A')}
Classification: {sample_info.get('class_name', 'N/A')}
Malignancy: {sample_info.get('malignancy_score', 'N/A')}

SEGMENTATION METRICS:
Dice: {metrics['dice']:.3f}
IoU: {metrics['iou']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}

GRADIENT STATISTICS:
Grad Range: [{np.min(grad_2d):.2e}, {np.max(grad_2d):.2e}]
IG Range: [{np.min(ig_2d):.2e}, {np.max(ig_2d):.2e}]

AREAS (pixels):
GT Area: {metrics['gt_area']}
Pred Area: {metrics['pred_area']}
Intersection: {metrics['intersection']}
"""
    axes[2, 5].text(0.05, 0.95, summary_text, transform=axes[2, 5].transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[2, 5].axis('off')
    
    # Set main title
    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    classification_label = sample_info.get("class_label", 0)
    class_name_display = class_names.get(classification_label, "Unknown")

    plt.suptitle(f'Comprehensive XAI Analysis - Patient {sample_info.get("patient_id", "N/A")} '
                f'({class_name_display})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_comprehensive_xai_analysis(unet_model, image_index_to_explain):
    """
    Run complete XAI analysis for a specific image.
    
    Args:
        unet_model: Trained U-Net model
        image_index_to_explain: Index of the image to analyze
    
    Returns:
        dict: Analysis results containing all computed explanations and data
    """
    print("="*80)
    print("COMPREHENSIVE U-NET XAI ANALYSIS")
    print("="*80)
    
    # Load data
    comprehensive_data, detailed_metadata, training_data = load_saved_data()
    
    # Get sample information
    sample_metadata, class_label, malignancy_score = get_sample_info(
        comprehensive_data, detailed_metadata, image_index_to_explain
    )
    
    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    
    # Prepare sample info dictionary
    sample_info = {
        'patient_id': sample_metadata['patient_id'],
        'uid': sample_metadata['uid'],
        'slice_idx': sample_metadata['slice_idx'],
        'class_label': class_label,
        'class_name': class_names.get(class_label, 'Unknown'),
        'malignancy_score': malignancy_score,
        'note': 'No segmentation GT available - using classification labels'
    }
    
    # Extract data for the specific image
    image_to_explain = comprehensive_data['X_test'][image_index_to_explain:image_index_to_explain+1]
    classification_label = comprehensive_data['y_test'][image_index_to_explain]
    unet_prediction = comprehensive_data['pred_masks'][image_index_to_explain:image_index_to_explain+1]

    # Get actual ground truth
    actual_ground_truth = get_actual_ground_truth_for_xai(sample_metadata)
    ground_truth = actual_ground_truth[np.newaxis, :512, :512, np.newaxis]

    print(f"Note: y_test contains classification labels, not segmentation masks.")
    print(f"Classification label for this sample: {classification_label}")

    print(f"\nData shapes:")
    print(f"  Image: {image_to_explain.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    print(f"  U-Net prediction: {unet_prediction.shape}")
    
    # Run gradient-based explanations
    print(f"\n1. COMPUTING GRADIENT EXPLANATION")
    print("-" * 40)
    try:
        gradients, _ = improved_gradient_explanation(unet_model, image_to_explain, 'mean_confidence')
        grad_success = True
    except Exception as e:
        print(f"Gradient explanation failed: {e}")
        gradients = None
        grad_success = False
    
    print(f"\n2. COMPUTING INTEGRATED GRADIENTS")
    print("-" * 40)
    try:
        integrated_grads, _ = integrated_gradients_explanation(unet_model, image_to_explain, steps=1000)
        ig_success = True
    except Exception as e:
        print(f"Integrated gradients failed: {e}")
        integrated_grads = None
        ig_success = False
    
    # Create comprehensive visualization
    print(f"\n3. CREATING COMPREHENSIVE VISUALIZATION")
    print("-" * 40)
    
    visualize_comprehensive_analysis(
        image=image_to_explain,
        ground_truth=ground_truth,
        unet_prediction=unet_prediction,
        gradients=gradients.numpy() if gradients is not None else None,
        integrated_grads=integrated_grads.numpy() if integrated_grads is not None else None,
        sample_info=sample_info,
        enhancement_factor=1000
    )
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Gradient explanation: {'Success' if grad_success else 'Failed'}")
    print(f"Integrated gradients: {'Success' if ig_success else 'Failed'}")
    print(f"Comprehensive visualization: Complete")
    
    return {
        'sample_info': sample_info,
        'gradients': gradients,
        'integrated_grads': integrated_grads,
        'image': image_to_explain,
        'ground_truth': ground_truth,
        'prediction': unet_prediction
    }


def main():
    """Main function to run XAI analysis"""
    # Initialize configuration
    Config.create_directories()
    
    # Load model
    model_path = '/workspace/output/models/best_unet_model.h5'
    unet_model = tf.keras.models.load_model(model_path, custom_objects={
        'focal_tversky_loss': focal_tversky_loss, 
        'dice_coefficient': dice_coefficient, 
        'dice_loss': dice_loss,
        'focal_loss': focal_loss,
        'iou': iou
    })
    
    # Example usage - analyze image at index 90
    image_index_to_explain = 90
    
    # Run the complete analysis
    results = run_comprehensive_xai_analysis(unet_model, image_index_to_explain)
    
    print("XAI analysis completed successfully!")
    return results


if __name__ == "__main__":
    # Set memory growth for GPU
    tf.keras.backend.clear_session()
    gc.collect()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    results = main()
    print("XAI analysis completed successfully!")
