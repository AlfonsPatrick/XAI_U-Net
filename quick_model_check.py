#!/usr/bin/env python3
"""
Quick check of the model to see if it's the right one and working properly
"""

import numpy as np
import tensorflow as tf
import pickle

def load_model():
    """Load the model with custom objects"""
    
    def focal_tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, gamma=0.75, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        focal_tversky = tf.pow((1 - tversky), gamma)
        return focal_tversky
    
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    def dice_loss(y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)
    
    def iou(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
        iou_score = (intersection + smooth) / (union + smooth)
        return iou_score
    
    def precision(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        predicted_pos = tf.reduce_sum(y_pred_flat)
        precision_score = (true_pos + smooth) / (predicted_pos + smooth)
        return precision_score
    
    def recall(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        actual_pos = tf.reduce_sum(y_true_flat)
        recall_score = (true_pos + smooth) / (actual_pos + smooth)
        return recall_score
    
    custom_objects = {
        'focal_tversky_loss': focal_tversky_loss,
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }
    
    model_path = "/workspace/output/models/best_unet_model.h5"
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def main():
    print("="*60)
    print("QUICK MODEL AND DATA CHECK")
    print("="*60)
    
    # Load model
    print("Loading model...")
    model = load_model()
    print("✓ Model loaded successfully")
    
    # Check model architecture
    print(f"\nModel input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Load data
    print("\nLoading data...")
    data = np.load("/workspace/output/comprehensive_results.npz", allow_pickle=True)
    training_data = np.load("/workspace/output/training_data_with_metadata.npz", allow_pickle=True)
    
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"Predictions shape: {data['pred_masks'].shape}")
    print(f"Ground truth shape: {training_data['m_test'].shape}")
    print(f"Classification labels: {data['y_test'].shape}")
    
    # Quick test prediction
    print("\nTesting model prediction...")
    test_image = data['X_test'][0:1]  # First test image
    prediction = model.predict(test_image, verbose=0)
    
    print(f"Test prediction shape: {prediction.shape}")
    print(f"Prediction range: [{np.min(prediction):.6f}, {np.max(prediction):.6f}]")
    print(f"Prediction mean: {np.mean(prediction):.6f}")
    print(f"Pixels > 0.5: {np.sum(prediction > 0.5)}")
    print(f"Pixels > 0.1: {np.sum(prediction > 0.1)}")
    
    # Check stored predictions vs fresh predictions
    stored_pred = data['pred_masks'][0]
    fresh_pred = prediction[0]
    
    print(f"\nComparing stored vs fresh predictions:")
    print(f"Stored prediction range: [{np.min(stored_pred):.6f}, {np.max(stored_pred):.6f}]")
    print(f"Fresh prediction range: [{np.min(fresh_pred):.6f}, {np.max(fresh_pred):.6f}]")
    print(f"Predictions match: {np.allclose(stored_pred, fresh_pred, atol=1e-6)}")
    
    # Check ground truth
    gt_mask = training_data['m_test'][0]
    print(f"\nGround truth analysis:")
    print(f"GT range: [{np.min(gt_mask):.6f}, {np.max(gt_mask):.6f}]")
    print(f"GT pixels > 0.5: {np.sum(gt_mask > 0.5)}")
    print(f"GT has nodule: {np.any(gt_mask > 0.5)}")
    
    # Quick metrics calculation
    if np.any(gt_mask > 0.5) or np.any(fresh_pred > 0.5):
        intersection = np.sum((fresh_pred > 0.5) * (gt_mask > 0.5))
        union_dice = np.sum(fresh_pred > 0.5) + np.sum(gt_mask > 0.5)
        union_iou = np.sum(fresh_pred > 0.5) + np.sum(gt_mask > 0.5) - intersection
        
        dice = (2.0 * intersection) / union_dice if union_dice > 0 else 0.0
        iou = intersection / union_iou if union_iou > 0 else 0.0
        
        print(f"\nSample metrics for first image:")
        print(f"Dice: {dice:.4f}")
        print(f"IoU: {iou:.4f}")
    
    # Check a few more samples
    print(f"\nChecking ground truth distribution:")
    gt_positive_samples = 0
    pred_positive_samples = 0
    
    for i in range(min(50, len(training_data['m_test']))):  # Check first 50 samples
        if np.any(training_data['m_test'][i] > 0.5):
            gt_positive_samples += 1
        if np.any(data['pred_masks'][i] > 0.5):
            pred_positive_samples += 1
    
    print(f"Ground truth positive samples (first 50): {gt_positive_samples}/50")
    print(f"Prediction positive samples (first 50): {pred_positive_samples}/50")
    
    # Classification analysis
    y_test = data['y_test']
    unique_labels, counts = np.unique(y_test, return_counts=True)
    print(f"\nClassification label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples")
    
    print("\n" + "="*60)
    print("QUICK DIAGNOSIS:")
    
    if gt_positive_samples == 0:
        print("🚨 ISSUE: No positive ground truth samples found!")
        print("   This explains the low metrics.")
    elif pred_positive_samples == 0:
        print("🚨 ISSUE: Model produces no positive predictions!")
        print("   Model might need different threshold or retraining.")
    elif gt_positive_samples < 5:
        print("⚠️  WARNING: Very few positive ground truth samples.")
        print("   This makes evaluation difficult.")
    else:
        print("✓ Both ground truth and predictions have positive samples.")
        print("  Low metrics might be due to poor overlap or challenging data.")
    
    print("="*60)

if __name__ == "__main__":
    main()