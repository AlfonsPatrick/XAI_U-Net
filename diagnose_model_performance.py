#!/usr/bin/env python3
"""
Diagnostic script to investigate why IoU and Dice scores are low
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all available data for diagnosis"""
    print("Loading data for diagnosis...")
    
    # Load comprehensive results
    data_path = "/workspace/output/comprehensive_results.npz"
    data = np.load(data_path, allow_pickle=True)
    
    # Load training data with metadata
    training_data_path = "/workspace/output/training_data_with_metadata.npz"
    training_data = np.load(training_data_path, allow_pickle=True)
    
    # Load metadata
    metadata_path = "/workspace/output/detailed_metadata.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return data, training_data, metadata

def analyze_ground_truth_vs_predictions(data, training_data):
    """Analyze the relationship between ground truth and predictions"""
    print("\n" + "="*80)
    print("ANALYZING GROUND TRUTH VS PREDICTIONS")
    print("="*80)
    
    X_test = data['X_test']
    pred_masks = data['pred_masks']
    ground_truth = training_data['m_test']
    
    print(f"Data shapes:")
    print(f"  X_test: {X_test.shape}")
    print(f"  Predictions: {pred_masks.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    # Check if shapes match
    if pred_masks.shape != ground_truth.shape:
        print("⚠️  WARNING: Shape mismatch between predictions and ground truth!")
        return
    
    # Analyze ground truth statistics
    gt_stats = {
        'num_positive': np.sum(ground_truth > 0.5),
        'num_samples_with_nodules': np.sum([np.any(mask > 0.5) for mask in ground_truth]),
        'mean_positive_pixels': np.mean([np.sum(mask > 0.5) for mask in ground_truth]),
        'max_positive_pixels': np.max([np.sum(mask > 0.5) for mask in ground_truth])
    }
    
    # Analyze prediction statistics
    pred_stats = {
        'num_positive': np.sum(pred_masks > 0.5),
        'num_samples_with_predictions': np.sum([np.any(mask > 0.5) for mask in pred_masks]),
        'mean_positive_pixels': np.mean([np.sum(mask > 0.5) for mask in pred_masks]),
        'max_positive_pixels': np.max([np.sum(mask > 0.5) for mask in pred_masks])
    }
    
    print(f"\nGround Truth Statistics:")
    print(f"  Samples with nodules: {gt_stats['num_samples_with_nodules']}/{len(ground_truth)}")
    print(f"  Total positive pixels: {gt_stats['num_positive']}")
    print(f"  Mean positive pixels per sample: {gt_stats['mean_positive_pixels']:.1f}")
    print(f"  Max positive pixels in a sample: {gt_stats['max_positive_pixels']}")
    
    print(f"\nPrediction Statistics:")
    print(f"  Samples with predictions: {pred_stats['num_samples_with_predictions']}/{len(pred_masks)}")
    print(f"  Total positive pixels: {pred_stats['num_positive']}")
    print(f"  Mean positive pixels per sample: {pred_stats['mean_positive_pixels']:.1f}")
    print(f"  Max positive pixels in a sample: {pred_stats['max_positive_pixels']}")
    
    # Check for potential issues
    if gt_stats['num_samples_with_nodules'] == 0:
        print("🚨 CRITICAL: No positive samples in ground truth!")
    elif gt_stats['num_samples_with_nodules'] < 10:
        print("⚠️  WARNING: Very few positive samples in ground truth")
    
    if pred_stats['num_samples_with_predictions'] == 0:
        print("🚨 CRITICAL: No positive predictions!")
    elif pred_stats['num_samples_with_predictions'] < 10:
        print("⚠️  WARNING: Very few positive predictions")
    
    return gt_stats, pred_stats

def visualize_sample_comparisons(data, training_data, num_samples=6):
    """Visualize sample comparisons between ground truth and predictions"""
    print(f"\nCreating sample comparisons...")
    
    X_test = data['X_test']
    pred_masks = data['pred_masks']
    ground_truth = training_data['m_test']
    
    # Find samples with ground truth nodules
    samples_with_gt = []
    for i in range(len(ground_truth)):
        if np.any(ground_truth[i] > 0.5):
            samples_with_gt.append(i)
    
    # Find samples with predictions
    samples_with_pred = []
    for i in range(len(pred_masks)):
        if np.any(pred_masks[i] > 0.5):
            samples_with_pred.append(i)
    
    print(f"Samples with ground truth nodules: {len(samples_with_gt)}")
    print(f"Samples with predictions: {len(samples_with_pred)}")
    
    # Select diverse samples for visualization
    if len(samples_with_gt) > 0:
        selected_indices = samples_with_gt[:num_samples//2]
    else:
        selected_indices = []
    
    if len(samples_with_pred) > 0:
        selected_indices.extend(samples_with_pred[:num_samples//2])
    
    # Fill with random samples if needed
    while len(selected_indices) < num_samples:
        idx = np.random.randint(0, len(X_test))
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    selected_indices = selected_indices[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        # Original image
        axes[i, 0].imshow(X_test[idx, :, :, 0], cmap='gray')
        axes[i, 0].set_title(f'Original {idx}')
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_sum = np.sum(ground_truth[idx] > 0.5)
        axes[i, 1].imshow(ground_truth[idx, :, :, 0], cmap='hot')
        axes[i, 1].set_title(f'Ground Truth\nPixels: {gt_sum}')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_sum = np.sum(pred_masks[idx] > 0.5)
        axes[i, 2].imshow(pred_masks[idx, :, :, 0], cmap='hot')
        axes[i, 2].set_title(f'Prediction\nPixels: {pred_sum}')
        axes[i, 2].axis('off')
        
        # Overlay comparison
        axes[i, 3].imshow(X_test[idx, :, :, 0], cmap='gray', alpha=0.5)
        axes[i, 3].imshow(ground_truth[idx, :, :, 0], cmap='Reds', alpha=0.3, label='GT')
        axes[i, 3].imshow(pred_masks[idx, :, :, 0], cmap='Blues', alpha=0.3, label='Pred')
        axes[i, 3].set_title('GT(Red) vs Pred(Blue)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('ground_truth_vs_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_metrics_with_different_thresholds(data, training_data):
    """Test different thresholds to see if that improves metrics"""
    print(f"\nTesting different thresholds...")
    
    pred_masks = data['pred_masks']
    ground_truth = training_data['m_test']
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for threshold in thresholds:
        pred_binary = (pred_masks > threshold).astype(np.float32)
        
        # Compute Dice and IoU for this threshold
        dice_scores = []
        iou_scores = []
        
        for i in range(len(pred_binary)):
            pred = pred_binary[i].flatten()
            true = ground_truth[i].flatten()
            
            intersection = np.sum(pred * true)
            union_dice = np.sum(pred) + np.sum(true)
            union_iou = np.sum(pred) + np.sum(true) - intersection
            
            if union_dice > 0:
                dice = (2.0 * intersection) / union_dice
            else:
                dice = 1.0 if np.sum(pred) == 0 else 0.0
            
            if union_iou > 0:
                iou = intersection / union_iou
            else:
                iou = 1.0 if np.sum(pred) == 0 else 0.0
            
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        
        results.append({
            'threshold': threshold,
            'dice': mean_dice,
            'iou': mean_iou
        })
        
        print(f"  Threshold {threshold:.1f}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")
    
    # Find best threshold
    best_dice = max(results, key=lambda x: x['dice'])
    best_iou = max(results, key=lambda x: x['iou'])
    
    print(f"\nBest Dice: {best_dice['dice']:.4f} at threshold {best_dice['threshold']}")
    print(f"Best IoU: {best_iou['iou']:.4f} at threshold {best_iou['threshold']}")
    
    return results

def check_data_consistency(data, training_data, metadata):
    """Check if the data splits are consistent"""
    print(f"\nChecking data consistency...")
    
    # Check if test UIDs match
    if 'test_metadata' in metadata:
        test_metadata = metadata['test_metadata']
        
        # Get UIDs from metadata
        metadata_uids = [sample.get('uid', '') for sample in test_metadata]
        
        # Check if we have the same number of samples
        print(f"Test samples in comprehensive_results: {len(data['X_test'])}")
        print(f"Test samples in training_data: {len(training_data['m_test'])}")
        print(f"Test samples in metadata: {len(test_metadata)}")
        
        if len(data['X_test']) != len(training_data['m_test']):
            print("🚨 CRITICAL: Mismatch in number of test samples!")
            return False
        
        # Check classification labels
        y_test_comprehensive = data['y_test']
        
        if 'y_test' in training_data:
            y_test_training = training_data['y_test']
            
            if not np.array_equal(y_test_comprehensive, y_test_training):
                print("⚠️  WARNING: Classification labels don't match between datasets")
                print(f"Comprehensive y_test unique values: {np.unique(y_test_comprehensive)}")
                print(f"Training y_test unique values: {np.unique(y_test_training)}")
            else:
                print("✓ Classification labels match between datasets")
        
        return True
    
    return False

def main():
    """Main diagnostic function"""
    print("="*80)
    print("DIAGNOSING MODEL PERFORMANCE ISSUES")
    print("="*80)
    
    try:
        # Load data
        data, training_data, metadata = load_data()
        
        # Check data consistency
        consistency_ok = check_data_consistency(data, training_data, metadata)
        
        if not consistency_ok:
            print("Data consistency issues detected!")
        
        # Analyze ground truth vs predictions
        gt_stats, pred_stats = analyze_ground_truth_vs_predictions(data, training_data)
        
        # Test different thresholds
        threshold_results = compute_metrics_with_different_thresholds(data, training_data)
        
        # Create visualizations
        visualize_sample_comparisons(data, training_data, num_samples=6)
        
        print("\n" + "="*80)
        print("DIAGNOSIS SUMMARY")
        print("="*80)
        
        # Provide diagnosis
        if gt_stats['num_samples_with_nodules'] == 0:
            print("🚨 MAIN ISSUE: No positive samples in ground truth!")
            print("   This suggests the ground truth masks are empty or incorrectly loaded.")
        elif pred_stats['num_samples_with_predictions'] == 0:
            print("🚨 MAIN ISSUE: Model produces no positive predictions!")
            print("   This suggests the model is not working properly or needs different thresholds.")
        elif gt_stats['num_samples_with_nodules'] < pred_stats['num_samples_with_predictions'] * 0.1:
            print("⚠️  ISSUE: Major imbalance between ground truth and predictions")
            print("   Ground truth might be too sparse or predictions too liberal.")
        else:
            print("✓ Data seems reasonable. Low scores might be due to:")
            print("  - Challenging dataset (medical segmentation is hard)")
            print("  - Model needs more training")
            print("  - Different preprocessing between training and testing")
        
        # Find best threshold
        best_result = max(threshold_results, key=lambda x: x['dice'])
        print(f"\n💡 RECOMMENDATION: Use threshold {best_result['threshold']} for best Dice score ({best_result['dice']:.4f})")
        
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()