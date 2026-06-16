"""
LIDC-IDRI Lung Nodule Detection and Classification System

This script implements a U-Net based lung nodule segmentation system with 
classification capabilities using the LIDC-IDRI dataset.

Configuration: 50 epochs, 100 patients, learning rate 5e-4, dice_weight 0.6,
dropout_rate=0.5, added mask preprocessing, min_size=100, increased max samples
per class by 50%, added batch processing, random seed=73, changed handcrafted
features to deep features, added focal tversky loss with alpha=0.4 beta=0.6,
added elastic deformation augmentation.
"""

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

# visualization
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.callbacks import Print
import albumentations as A

print("All imports completed successfully!")


class Config:
    """Configuration class for the lung nodule detection system"""
    DATA_DIR = "/workspace/storage/lidc_dataset"
    METADATA_PATH = "workspace/storage/LIDC-IDRI_MetaData.csv"
    OUTPUT_DIR = "workspace/output"
    
    # Image processing parameters
    IMAGE_SIZE = (512, 512)
    PATCH_SIZE = 32
    STRIDE = 32
    HU_MIN = -1000
    HU_MAX = 400
    
    # Training parameters
    BATCH_SIZE = 8
    PATCH_BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 5e-4
    VALIDATION_SPLIT = 0.2
    
    # Model parameters
    UNET_FILTERS = [32, 64, 128, 256]
    DROPOUT_RATE = 0.5
    
    # Data balance parameters
    NUM_PATIENTS = 100
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


def find_nodule_containing_slices(nodule_data_dict, slice_range=5):
    """Find slices that contain nodules and their surrounding context"""
    nodule_slice_indices = []
    
    for uid, nodule_list in nodule_data_dict.items():
        if nodule_list:
            nodule_slice_indices.append(uid)
    
    return nodule_slice_indices


def extract_balanced_dataset_with_metadata(patient_data_list, max_samples_per_class=None):
    """Enhanced dataset extraction that tracks metadata for post-processing"""
    if max_samples_per_class is None:
        max_samples_per_class = {'normal': 400, 'benign': 300, 'malignant': 300}
    
    class_samples = {'normal': [], 'benign': [], 'malignant': []}
    sample_metadata = []
    
    for patient_id, dicom_data, nodule_data in patient_data_list:
        print(f"Processing patient {patient_id}...")
        
        # Create UID to index mapping
        uid_to_idx = {data['uid']: idx for idx, data in enumerate(dicom_data)}
        
        # Find slices with nodules
        nodule_slice_indices = set()
        for uid in nodule_data.keys():
            if uid in uid_to_idx:
                idx = uid_to_idx[uid]
                # Add surrounding slices for context
                for offset in range(-Config.NODULE_CONTEXT_SLICES, Config.NODULE_CONTEXT_SLICES + 1):
                    context_idx = idx + offset
                    if 0 <= context_idx < len(dicom_data):
                        nodule_slice_indices.add(context_idx)
        
        # Process each slice
        for idx, slice_data in enumerate(dicom_data):
            image = slice_data['image']
            uid = slice_data['uid']
            
            # Normalize and resize image
            image_norm = normalize_hu(image)
            image_resized = cv2.resize(image_norm, Config.IMAGE_SIZE)
            image_resized = np.expand_dims(image_resized, axis=-1)
            
            # Create mask
            if uid in nodule_data:
                mask, malignancy = create_mask_from_nodule_data(nodule_data[uid], image.shape)
                mask_resized = cv2.resize(mask, Config.IMAGE_SIZE)
                mask_resized = np.expand_dims(mask_resized, axis=-1)
                
                # Classify based on malignancy
                if malignancy == 0:
                    class_label = 'normal'
                elif 1 <= malignancy <= Config.MALIGNANCY_THRESHOLD:
                    class_label = 'benign'
                else:
                    class_label = 'malignant'
                
            else:
                # Normal slice (no nodules)
                if idx in nodule_slice_indices:
                    class_label = 'normal'
                else:
                    if len(class_samples['normal']) >= max_samples_per_class['normal']:
                        continue
                    class_label = 'normal'
                
                mask_resized = np.zeros((*Config.IMAGE_SIZE, 1), dtype=np.uint8)
                malignancy = 0
            
            # Add to appropriate class with sampling limits
            if len(class_samples[class_label]) < max_samples_per_class[class_label]:
                class_samples[class_label].append({
                    'image': image_resized,
                    'mask': mask_resized,
                    'malignancy': malignancy,
                    'patient_id': patient_id,
                    'slice_idx': idx
                })
                
                # Track metadata for each sample
                sample_metadata.append({
                    'patient_id': patient_id,
                    'slice_idx': idx,
                    'uid': uid,
                    'class_label': class_label,
                    'malignancy': malignancy,
                    'original_shape': image.shape,
                    'has_nodule': uid in nodule_data,
                    'is_context_slice': idx in nodule_slice_indices and uid not in nodule_data
                })
    
    # Convert to arrays
    all_images, all_masks, all_labels = [], [], []
    final_metadata = []
    
    for class_name, samples in class_samples.items():
        print(f"{class_name}: {len(samples)} samples")
        
        for sample in samples:
            all_images.append(sample['image'])
            all_masks.append(sample['mask'])
            
            # Convert class name to numeric label
            if class_name == 'normal':
                all_labels.append(0)
            elif class_name == 'benign':
                all_labels.append(1)
            else:  # malignant
                all_labels.append(2)
            
            # Find corresponding metadata
            matching_metadata = next((m for m in sample_metadata 
                                    if m['patient_id'] == sample['patient_id'] 
                                    and m['slice_idx'] == sample['slice_idx']), None)
            if matching_metadata:
                matching_metadata['final_sample_index'] = len(final_metadata)
                final_metadata.append(matching_metadata)
    
    return np.array(all_images), np.array(all_masks), np.array(all_labels), final_metadata


def post_process_mask(mask, min_size=100):
    """Clean up binary segmentation mask by removing small objects"""
    mask_bool = mask.astype(bool)
    cleaned_mask = remove_small_objects(mask_bool, min_size=min_size)
    return cleaned_mask.astype(np.uint8)


def get_train_augmentations(image_size):
    """Define augmentation pipeline using Albumentations"""
    return A.Compose([
        # Elastic transformation for tissue deformation simulation
        A.ElasticTransform(
            p=0.5,
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            border_mode=cv2.BORDER_CONSTANT
        ),
        # Other augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            p=0.5,
            shift_limit=0.06,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2)
    ])


def augment_data(images, masks, labels, augmentation_factor=2):
    """Apply data augmentation using Albumentations pipeline"""
    aug_images, aug_masks, aug_labels = [], [], []
    
    image_size = images.shape[1:3]
    augmenter = get_train_augmentations(image_size)

    print(f"Augmenting data (factor: {augmentation_factor}x)...")
    for i in tqdm(range(len(images))):
        image = images[i]
        mask = masks[i]
        label = labels[i]

        # Add original image and mask
        aug_images.append(image)
        aug_masks.append(mask)
        aug_labels.append(label)

        # Create augmented versions
        for _ in range(augmentation_factor):
            augmented = augmenter(image=image, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            
            aug_images.append(aug_img)
            aug_masks.append(aug_mask)
            aug_labels.append(label)
    
    return np.array(aug_images), np.array(aug_masks), np.array(aug_labels)


def save_training_data_with_metadata(X_train, X_val, X_test, m_train, m_val, m_test, 
                                   y_train, y_val, y_test, 
                                   train_metadata, val_metadata, test_metadata):
    """Save training data along with metadata for post-processing"""
    
    save_path = os.path.join(Config.OUTPUT_DIR, 'training_data_with_metadata.npz')
    np.savez_compressed(
        save_path,
        # Original data
        X_train=X_train, X_val=X_val, X_test=X_test,
        m_train=m_train, m_val=m_val, m_test=m_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        
        # Metadata arrays for easier access
        train_patient_ids=np.array([m['patient_id'] for m in train_metadata]),
        val_patient_ids=np.array([m['patient_id'] for m in val_metadata]),
        test_patient_ids=np.array([m['patient_id'] for m in test_metadata]),
        train_slice_indices=np.array([m['slice_idx'] for m in train_metadata]),
        val_slice_indices=np.array([m['slice_idx'] for m in val_metadata]),
        test_slice_indices=np.array([m['slice_idx'] for m in test_metadata]),
        train_uids=np.array([m['uid'] for m in train_metadata]),
        val_uids=np.array([m['uid'] for m in val_metadata]),
        test_uids=np.array([m['uid'] for m in test_metadata])
    )
    
    print(f"Training data with metadata saved to: {save_path}")


def create_enhanced_unet(input_shape=(256, 256, 1), num_classes=1):
    """Create enhanced U-Net with L2 regularization and attention mechanism"""
    
    def conv_block(x, filters, dropout_rate=0.2):
        """Convolutional block with batch normalization, dropout, and L2 regularization"""
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
    
    def attention_block(x, g, filters):
        """Attention mechanism for U-Net"""
        x1 = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        
        g1 = tf.keras.layers.Conv2D(filters, 1, padding='same')(g)
        g1 = tf.keras.layers.BatchNormalization()(g1)
        
        psi = tf.keras.layers.Add()([x1, g1])
        psi = tf.keras.layers.Activation('relu')(psi)
        psi = tf.keras.layers.Conv2D(1, 1, padding='same')(psi)
        psi = tf.keras.layers.BatchNormalization()(psi)
        psi = tf.keras.layers.Activation('sigmoid')(psi)
        
        return tf.keras.layers.Multiply()([x, psi])
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = conv_block(inputs, 32, Config.DROPOUT_RATE)
    pool1 = tf.keras.layers.MaxPooling2D(2)(conv1)
    
    conv2 = conv_block(pool1, 64, Config.DROPOUT_RATE)
    pool2 = tf.keras.layers.MaxPooling2D(2)(conv2)
    
    conv3 = conv_block(pool2, 128, Config.DROPOUT_RATE)
    pool3 = tf.keras.layers.MaxPooling2D(2)(conv3)
    
    conv4 = conv_block(pool3, 256, Config.DROPOUT_RATE)
    pool4 = tf.keras.layers.MaxPooling2D(2)(conv4)
    
    # Bridge
    conv5 = conv_block(pool4, 512, Config.DROPOUT_RATE)
    
    # Decoder with attention
    up6 = tf.keras.layers.UpSampling2D(2)(conv5)
    att6 = attention_block(conv4, up6, 256)
    concat6 = tf.keras.layers.Concatenate()([up6, att6])
    conv6 = conv_block(concat6, 256, Config.DROPOUT_RATE)
    
    up7 = tf.keras.layers.UpSampling2D(2)(conv6)
    att7 = attention_block(conv3, up7, 128)
    concat7 = tf.keras.layers.Concatenate()([up7, att7])
    conv7 = conv_block(concat7, 128, Config.DROPOUT_RATE)
    
    up8 = tf.keras.layers.UpSampling2D(2)(conv7)
    att8 = attention_block(conv2, up8, 64)
    concat8 = tf.keras.layers.Concatenate()([up8, att8])
    conv8 = conv_block(concat8, 64, Config.DROPOUT_RATE)
    
    up9 = tf.keras.layers.UpSampling2D(2)(conv8)
    att9 = attention_block(conv1, up9, 32)
    concat9 = tf.keras.layers.Concatenate()([up9, att9])
    conv9 = conv_block(concat9, 32, Config.DROPOUT_RATE)
    
    # Output layer
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid', name='segmentation')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Compute Dice coefficient"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)


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


def plot_fixed_confusion_matrix(cm, class_names, ax, title='Confusion Matrix'):
    """Plot confusion matrix with manual text annotation to bypass clipping bug"""
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Manual text annotation fix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)


def main():
    """Enhanced main execution function with metadata tracking"""
    print("Starting enhanced LIDC-IDRI lung nodule detection and classification...")
    tf.keras.backend.clear_session()
    
    # Initialize configuration
    Config.create_directories()
    
    # Load data
    print("Loading patient data...")
    patient_data_list = []
    random.seed(73) 
    all_patient_ids = random.sample(range(1, 601), k=600)
    selected_patient_ids = all_patient_ids[:Config.NUM_PATIENTS]

    for patient_id in tqdm(selected_patient_ids, desc="Processing patients"):
        patient_dir = patient_root(patient_id)
        if not os.path.exists(patient_dir): 
            continue
        
        dicom_files = [os.path.join(r, f) for r, _, files in os.walk(patient_dir) 
                      for f in files if f.endswith('.dcm')]
        if not dicom_files: 
            continue
        
        dicom_data = load_dicom_series_enhanced(dicom_files)
        if not dicom_data: 
            continue
        
        xml_files = [os.path.join(r, f) for r, _, files in os.walk(patient_dir) 
                    for f in files if f.endswith('.xml')]
        if not xml_files: 
            continue
        
        nodule_data = {}
        for xml_file in xml_files:
            nodule_data.update(parse_lidc_xml_enhanced(xml_file))
        
        patient_data_list.append((patient_id, dicom_data, nodule_data))
    
    print(f"Successfully loaded data for {len(patient_data_list)} patients")
    
    if len(patient_data_list) == 0:
        raise ValueError(
            f"No patient data could be loaded. Please ensure your LIDC-IDRI dataset "
            f"is located at '{Config.DATA_DIR}' and follows the correct folder structure."
        )

    # Extract balanced dataset with metadata
    print("Extracting balanced dataset with metadata...")
    images, masks, labels, metadata = extract_balanced_dataset_with_metadata(patient_data_list)
    print(f"Dataset summary: Total={len(images)}, Normal={np.sum(labels == 0)}, "
          f"Benign={np.sum(labels == 1)}, Malignant={np.sum(labels == 2)}")
    
    # Apply data augmentation
    print("Applying data augmentation...")
    aug_images, aug_masks, aug_labels = augment_data(images, masks, labels)
    
    # Duplicate metadata for augmented samples
    augmentation_factor = 2
    aug_metadata = []
    for i, meta in enumerate(metadata):
        aug_metadata.append(meta.copy())
        for aug_idx in range(augmentation_factor):
            aug_meta = meta.copy()
            aug_meta['is_augmented'] = True
            aug_meta['augmentation_idx'] = aug_idx
            aug_meta['original_sample_idx'] = i
            aug_metadata.append(aug_meta)
    
    # Clean up memory
    del images, masks, labels, patient_data_list, metadata
    gc.collect()

    # Split data
    print("Splitting data...")
    total_samples = len(aug_images)
    all_indices = np.arange(total_samples)
    
    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, stratify=aug_labels, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, stratify=aug_labels[temp_indices], random_state=42
    )
    
    # Split data using indices
    X_train = aug_images[train_indices]
    X_val = aug_images[val_indices]
    X_test = aug_images[test_indices]
    y_train = aug_labels[train_indices]
    y_val = aug_labels[val_indices]
    y_test = aug_labels[test_indices]
    m_train = aug_masks[train_indices]
    m_val = aug_masks[val_indices]
    m_test = aug_masks[test_indices]
    
    # Split metadata
    train_metadata = [aug_metadata[i] for i in train_indices]
    val_metadata = [aug_metadata[i] for i in val_indices]
    test_metadata = [aug_metadata[i] for i in test_indices]
    
    # Clean up augmented data
    del aug_images, aug_masks, aug_labels, aug_metadata
    gc.collect()
    
    print(f"Training set: {len(X_train)} | Validation set: {len(X_val)} | Test set: {len(X_test)}")
    
    # Save training data with metadata
    print("Saving training data with metadata...")
    save_training_data_with_metadata(X_train, X_val, X_test, m_train, m_val, m_test, 
                                   y_train, y_val, y_test, 
                                   train_metadata, val_metadata, test_metadata)
    
    # U-Net Training
    print("Creating or loading U-Net model...")
    
    # Calculate class weights for segmentation loss
    total_pixels = m_train.size
    nodule_pixels = np.sum(m_train)
    background_pixels = total_pixels - nodule_pixels
    weight_for_0 = (total_pixels / (2 * background_pixels))
    weight_for_1 = (total_pixels / (2 * nodule_pixels))
    print(f"Weights for segmentation loss: Background={weight_for_0:.2f}, Nodule={weight_for_1:.2f}")
    
    # Load or create model
    history_path = os.path.join(Config.OUTPUT_DIR, 'results', 'training_history.json')
    
    if os.path.exists(history_path):
        print("Found existing training history. Loading to continue...")
        try:
            with open(history_path, 'r') as f:
                full_history = json.load(f)
        except json.JSONDecodeError:
            print("Warning: History file was corrupt. Starting fresh.")
            full_history = {}
    else:
        full_history = {}

    model_save_path = os.path.join(Config.OUTPUT_DIR, 'models', 'best_unet_model.h5')
    
    # Learning rate schedule
    initial_learning_rate = Config.LEARNING_RATE
    decay_steps = Config.EPOCHS * (len(X_train) // Config.BATCH_SIZE)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps=decay_steps, alpha=0.0
    )

    # Search for existing model
    model_load_path = None
    for dirname, _, filenames in os.walk('/workspace/output/'):
        if 'best_unet_model.h5' in filenames:
            potential_path = os.path.join(dirname, 'best_unet_model.h5')
            if os.path.basename(os.path.dirname(potential_path)) == 'models':
                model_load_path = potential_path
                break

    # Load or create model
    if model_load_path:
        print(f"Found existing model at {model_load_path}. Loading...")
        custom_objects = {
            'focal_tversky_loss': focal_tversky_loss, 
            'dice_coefficient': dice_coefficient, 
            'dice_loss': dice_loss,
            'iou': iou
        }
        unet_model = tf.keras.models.load_model(model_load_path, custom_objects=custom_objects)
    else:
        print("Creating new U-Net model...")
        unet_model = create_enhanced_unet(input_shape=(*Config.IMAGE_SIZE, 1))
    
    # Compile model
    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss=focal_tversky_loss,
        metrics=[dice_coefficient, dice_loss, iou, 
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25, mode="min", 
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, 
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_loss', 
            save_best_only=True, verbose=1
        )
    ]
    
    # Train model
    history = unet_model.fit(
        X_train, m_train, validation_data=(X_val, m_val),
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, 
        callbacks=callbacks, verbose=1
    )
    
    # Save training history
    for key, value in history.history.items():
        if key in full_history:
            full_history[key].extend(value)
        else:
            full_history[key] = value

    serializable_history = {}
    for key, value in full_history.items():
        serializable_history[key] = [float(v) for v in value]

    with open(history_path, 'w') as f:
        json.dump(serializable_history, f)
    
    print("Training history saved.")

    # Generate predictions and save results
    print("Generating test predictions...")
    test_predictions_masks = unet_model.predict(X_test)
    
    # Save comprehensive results
    print("Saving comprehensive results...")
    results_path = os.path.join(Config.OUTPUT_DIR, 'comprehensive_results.npz')
    np.savez_compressed(
        results_path,
        # Test data
        X_test=X_test, y_test=y_test, m_test=m_test, pred_masks=test_predictions_masks,
        # Training data
        X_train=X_train, y_train=y_train, m_train=m_train,
        # Validation data
        X_val=X_val, y_val=y_val, m_val=m_val,
        # Metadata arrays
        test_patient_ids=np.array([m['patient_id'] for m in test_metadata]),
        test_slice_indices=np.array([m['slice_idx'] for m in test_metadata]),
        test_uids=np.array([m['uid'] for m in test_metadata]),
        test_class_labels=np.array([m['class_label'] for m in test_metadata]),
        test_malignancy_scores=np.array([m['malignancy'] for m in test_metadata]),
        train_patient_ids=np.array([m['patient_id'] for m in train_metadata]),
        train_slice_indices=np.array([m['slice_idx'] for m in train_metadata]),
        train_uids=np.array([m['uid'] for m in train_metadata]),
        val_patient_ids=np.array([m['patient_id'] for m in val_metadata]),
        val_slice_indices=np.array([m['slice_idx'] for m in val_metadata]),
        val_uids=np.array([m['uid'] for m in val_metadata])
    )
    
    # Save detailed metadata
    metadata_save_path = os.path.join(Config.OUTPUT_DIR, 'detailed_metadata.pkl')
    with open(metadata_save_path, 'wb') as f:
        pickle.dump({
            'train_metadata': train_metadata,
            'val_metadata': val_metadata,
            'test_metadata': test_metadata,
            'selected_patient_ids': selected_patient_ids,
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'test_indices': test_indices.tolist()
        }, f)
    
    print(f"Results saved to: {results_path}")
    print(f"Detailed metadata saved to: {metadata_save_path}")

    # Visualization
    print("Generating visualizations...")

    # Plot training history
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('U-Net Training Metrics', fontsize=16)

    axes[0, 0].plot(full_history['loss'], label='Training Loss')
    axes[0, 0].plot(full_history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(full_history['dice_coefficient'], label='Training Dice')
    axes[0, 1].plot(full_history['val_dice_coefficient'], label='Validation Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].legend()

    axes[0, 2].plot(full_history['iou'], label='Training IoU')
    axes[0, 2].plot(full_history['val_iou'], label='Validation IoU')
    axes[0, 2].set_title('Intersection over Union (IoU)')
    axes[0, 2].legend()

    # Plot precision and recall
    precision_key = [k for k in full_history.keys() if 'precision' in k and 'val' not in k][0]
    val_precision_key = [k for k in full_history.keys() if 'precision' in k and 'val' in k][0]
    axes[1, 0].plot(full_history[precision_key], label='Training Precision')
    axes[1, 0].plot(full_history[val_precision_key], label='Validation Precision')
    axes[1, 0].set_title('Segmentation Precision')
    axes[1, 0].legend()

    recall_key = [k for k in full_history.keys() if 'recall' in k and 'val' not in k][0]
    val_recall_key = [k for k in full_history.keys() if 'recall' in k and 'val' in k][0]
    axes[1, 1].plot(full_history[recall_key], label='Training Recall')
    axes[1, 1].plot(full_history[val_recall_key], label='Validation Recall')
    axes[1, 1].set_title('Segmentation Recall')
    axes[1, 1].legend()
    
    axes[1, 2].plot(full_history['dice_loss'], label='Training Dice Loss')
    axes[1, 2].plot(full_history['val_dice_loss'], label='Validation Dice Loss')
    axes[1, 2].set_title('Dice Loss')
    axes[1, 2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'results', 'unet_training_history.png'), dpi=300)
    plt.show()

    # Visualize sample predictions
    print("Visualizing sample predictions...")
    sample_indices = np.random.choice(len(X_test), size=min(6, len(X_test)), replace=False)
    
    sample_images_to_plot = X_test[sample_indices]
    predicted_masks_for_plot = (test_predictions_masks[sample_indices] > Config.SEGMENTATION_THRESHOLD).astype(np.uint8)
    predicted_masks_for_plot = np.array([post_process_mask(m, min_size=100) for m in predicted_masks_for_plot])

    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    fig.suptitle('Sample Predictions vs. Ground Truth', fontsize=16)
    
    for i, idx in enumerate(sample_indices):
        axes[0, i].imshow(X_test[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nTrue: {["Normal", "Benign", "Malignant"][y_test[idx]]}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(m_test[idx].squeeze(), cmap='hot')
        axes[1, i].set_title('Ground Truth Mask')
        axes[1, i].axis('off')
        
        pred_mask = predicted_masks_for_plot[i].squeeze()
        axes[2, i].imshow(pred_mask, cmap='hot')
        axes[2, i].set_title('Predicted Mask')
        axes[2, i].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'results', 'sample_predictions.png'), dpi=300)
    plt.show()

    # Save final model
    print("Saving U-Net model...")
    unet_model.save(os.path.join(Config.OUTPUT_DIR, 'models', 'best_unet_model.h5'))

    print("U-Net training and evaluation completed successfully!")
    print("Files saved for post-processing:")
    print(f"  - Model results: {results_path}")
    print(f"  - Detailed metadata: {metadata_save_path}")
    
    return unet_model


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
    
    model = main()
    print("Model trained successfully!")
