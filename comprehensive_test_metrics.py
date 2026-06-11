#!/usr/bin/env python3
"""
Comprehensive Test Metrics for Attention U-Net

This script provides comprehensive testing metrics for the attention U-Net model,
including proper ground truth loading and evaluation on the entire test set.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
warnings.filterwarnings('ignore')

# Set up GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class ComprehensiveUNetTester:
    """Comprehensive tester for U-Net model with proper metrics"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.metadata = None
        self.ground_truth_masks = None
        
    def load_all_data(self):
        """Load model, data, and ground truth masks"""
        print("="*80)
        print("LOADING MODEL AND DATA")
        print("="*80)
        
        # Define custom objects for model loading
        custom_objects = {
            'focal_tversky_loss': self.focal_tversky_loss,
            'dice_coefficient': self.dice_coefficient,
            'dice_loss': self.dice_loss,
            'iou': self.iou,
            'precision': self.precision,
            'recall': self.recall
        }
        
        # Load the trained model
        model_path = "/workspace/output/models/best_unet_model.h5"
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✓ Model loaded successfully")
        
        # Load comprehensive data
        data_path = "/workspace/output/comprehensive_results.npz"
        print(f"Loading data from: {data_path}")
        self.data = np.load(data_path, allow_pickle=True)
        print("✓ Data loaded successfully")
        
        # Load metadata
        metadata_path = "/workspace/output/detailed_metadata.pkl"
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print("✓ Metadata loaded successfully")
        
        # Try to load ground truth masks
        self.ground_truth_masks = self.load_ground_truth_masks()
        
        # Print data shapes
        print(f"\nData shapes:")
        print(f"  X_test: {self.data['X_test'].shape}")
        print(f"  y_test (classification): {self.data['y_test'].shape}")
        print(f"  pred_masks: {self.data['pred_masks'].shape}")
        if self.ground_truth_masks is not None:
            print(f"  Ground truth masks: {self.ground_truth_masks.shape}")
        else:
            print("  Ground truth masks: Not available")
    
    def load_ground_truth_masks(self):
        """Load ground truth segmentation masks"""
        print("\nAttempting to load ground truth masks...")
        
        # Try to load from training data with metadata
        training_data_path = "/workspace/output/training_data_with_metadata.npz"
        if os.path.exists(training_data_path):
            try:
                training_data = np.load(training_data_path, allow_pickle=True)
                if 'm_test' in training_data:
                    print("✓ Found ground truth masks in training data")
                    return training_data['m_test']
            except Exception as e:
                print(f"Could not load from training data: {e}")
        
        # Try alternative data file
        alt_data_path = "/workspace/output_latest/data_splits.npz"
        if os.path.exists(alt_data_path):
            try:
                alt_data = np.load(alt_data_path, allow_pickle=True)
                if 'm_test' in alt_data:
                    print("✓ Found ground truth masks in alternative data file")
                    return alt_data['m_test']
            except Exception as e:
                print(f"Could not load from alternative data: {e}")
        
        print("⚠ No ground truth masks found - will use predictions for analysis")
        return None
    
    @staticmethod
    def focal_tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, gamma=0.75, smooth=1e-6):
        """Focal Tversky Loss function"""
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
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    @staticmethod
    def dice_loss(y_true, y_pred):
        """Dice loss function"""
        return 1 - ComprehensiveUNetTester.dice_coefficient(y_true, y_pred)
    
    @staticmethod
    def iou(y_true, y_pred, smooth=1e-6):
        """Intersection over Union metric"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
        
        iou_score = (intersection + smooth) / (union + smooth)
        return iou_score
    
    @staticmethod
    def precision(y_true, y_pred, smooth=1e-6):
        """Precision metric"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        predicted_pos = tf.reduce_sum(y_pred_flat)
        
        precision_score = (true_pos + smooth) / (predicted_pos + smooth)
        return precision_score
    
    @staticmethod
    def recall(y_true, y_pred, smooth=1e-6):
        """Recall metric"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        actual_pos = tf.reduce_sum(y_true_flat)
        
        recall_score = (true_pos + smooth) / (actual_pos + smooth)
        return recall_score
    
    def compute_segmentation_metrics(self, predictions, ground_truth, threshold=0.5):
        """Compute comprehensive segmentation metrics"""
        print("\n" + "="*80)
        print("COMPUTING SEGMENTATION METRICS")
        print("="*80)
        
        # Threshold predictions
        pred_binary = (predictions > threshold).astype(np.float32)
        
        # Compute metrics for each sample
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        
        print("Computing metrics for each sample...")
        for i in range(len(predictions)):
            pred = pred_binary[i:i+1]
            true = ground_truth[i:i+1]
            
            dice_score = self.dice_coefficient(true, pred).numpy()
            iou_score = self.iou(true, pred).numpy()
            precision_score = self.precision(true, pred).numpy()
            recall_score = self.recall(true, pred).numpy()
            
            dice_scores.append(dice_score)
            iou_scores.append(iou_score)
            precision_scores.append(precision_score)
            recall_scores.append(recall_score)
        
        # Compute statistics
        metrics = {
            'dice_scores': dice_scores,
            'iou_scores': iou_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'median_dice': np.median(dice_scores),
            'mean_iou': np.mean(iou_scores),
            'std_iou': np.std(iou_scores),
            'median_iou': np.median(iou_scores),
            'mean_precision': np.mean(precision_scores),
            'std_precision': np.std(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'std_recall': np.std(recall_scores),
        }
        
        # Print results
        print(f"\nSegmentation Metrics (n={len(predictions)}):")
        print(f"  Dice Score:  {metrics['mean_dice']:.4f} ± {metrics['std_dice']:.4f} (median: {metrics['median_dice']:.4f})")
        print(f"  IoU Score:   {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f} (median: {metrics['median_iou']:.4f})")
        print(f"  Precision:   {metrics['mean_precision']:.4f} ± {metrics['std_precision']:.4f}")
        print(f"  Recall:      {metrics['mean_recall']:.4f} ± {metrics['std_recall']:.4f}")
        
        return metrics
    
    def compute_classification_metrics(self):
        """Compute classification metrics based on nodule presence"""
        print("\n" + "="*80)
        print("COMPUTING CLASSIFICATION METRICS")
        print("="*80)
        
        # Get classification labels
        y_true = self.data['y_test']
        
        # Compute predicted labels based on segmentation masks
        pred_masks = self.data['pred_masks']
        y_pred = []
        
        for mask in pred_masks:
            # If mask has significant activation, predict positive class
            mask_sum = np.sum(mask > 0.5)
            if mask_sum > 100:  # Threshold for minimum nodule size
                y_pred.append(1)  # Nodule present
            else:
                y_pred.append(0)  # No nodule
        
        y_pred = np.array(y_pred)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Nodule', 'Nodule Present']))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Nodule', 'Nodule Present'],
                   yticklabels=['No Nodule', 'Nodule Present'])
        plt.title('Confusion Matrix - Nodule Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'confusion_matrix': cm
        }
    
    def analyze_prediction_quality(self):
        """Analyze the quality of predictions"""
        print("\n" + "="*80)
        print("ANALYZING PREDICTION QUALITY")
        print("="*80)
        
        pred_masks = self.data['pred_masks']
        
        # Analyze prediction statistics
        mask_sums = [np.sum(mask) for mask in pred_masks]
        mask_maxes = [np.max(mask) for mask in pred_masks]
        mask_means = [np.mean(mask) for mask in pred_masks]
        
        # Count predictions with significant activation
        significant_predictions = sum(1 for s in mask_sums if s > 1000)
        
        print(f"Prediction Statistics:")
        print(f"  Total test samples: {len(pred_masks)}")
        print(f"  Predictions with significant activation: {significant_predictions}")
        print(f"  Mean mask sum: {np.mean(mask_sums):.2f} ± {np.std(mask_sums):.2f}")
        print(f"  Mean max value: {np.mean(mask_maxes):.4f} ± {np.std(mask_maxes):.4f}")
        print(f"  Mean average value: {np.mean(mask_means):.6f} ± {np.std(mask_means):.6f}")
        
        # Plot histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(mask_sums, bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of Mask Sums')
        axes[0, 0].set_xlabel('Sum of Mask Values')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(mask_maxes, bins=50, alpha=0.7)
        axes[0, 1].set_title('Distribution of Max Values')
        axes[0, 1].set_xlabel('Max Mask Value')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(mask_means, bins=50, alpha=0.7)
        axes[1, 0].set_title('Distribution of Mean Values')
        axes[1, 0].set_xlabel('Mean Mask Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # Plot sample predictions
        sample_indices = [0, 50, 100, 150, 200]
        for i, idx in enumerate(sample_indices[:5]):
            if i < 5:
                row = 1
                col = 1
                if i == 4:  # Last subplot
                    axes[row, col].imshow(pred_masks[idx][:, :, 0], cmap='hot')
                    axes[row, col].set_title(f'Sample Prediction {idx}')
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'mask_sums': mask_sums,
            'mask_maxes': mask_maxes,
            'mask_means': mask_means,
            'significant_predictions': significant_predictions
        }
    
    def create_sample_visualizations(self, num_samples=10):
        """Create visualizations for sample predictions"""
        print(f"\n" + "="*80)
        print(f"CREATING SAMPLE VISUALIZATIONS ({num_samples} samples)")
        print("="*80)
        
        X_test = self.data['X_test']
        pred_masks = self.data['pred_masks']
        y_test = self.data['y_test']
        
        # Select diverse samples
        total_samples = len(X_test)
        indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Original image
            axes[i, 0].imshow(X_test[idx, :, :, 0], cmap='gray')
            axes[i, 0].set_title(f'Original {idx}\nClass: {y_test[idx]}')
            axes[i, 0].axis('off')
            
            # Prediction mask
            axes[i, 1].imshow(pred_masks[idx, :, :, 0], cmap='hot')
            axes[i, 1].set_title(f'Prediction\nMax: {np.max(pred_masks[idx]):.3f}')
            axes[i, 1].axis('off')
            
            # Overlay
            axes[i, 2].imshow(X_test[idx, :, :, 0], cmap='gray', alpha=0.7)
            axes[i, 2].imshow(pred_masks[idx, :, :, 0], cmap='hot', alpha=0.5)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation of the model"""
        print("="*80)
        print("COMPREHENSIVE ATTENTION U-NET EVALUATION")
        print("="*80)
        
        # Load all data
        self.load_all_data()
        
        # Analyze prediction quality
        pred_analysis = self.analyze_prediction_quality()
        
        # Compute classification metrics
        class_metrics = self.compute_classification_metrics()
        
        # If ground truth masks are available, compute segmentation metrics
        if self.ground_truth_masks is not None:
            seg_metrics = self.compute_segmentation_metrics(
                self.data['pred_masks'], 
                self.ground_truth_masks
            )
        else:
            print("\n⚠ Skipping segmentation metrics - no ground truth masks available")
            seg_metrics = None
        
        # Create sample visualizations
        self.create_sample_visualizations(num_samples=8)
        
        # Summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"✓ Model successfully loaded and evaluated")
        print(f"✓ Test set size: {len(self.data['X_test'])} samples")
        print(f"✓ Predictions with significant activation: {pred_analysis['significant_predictions']}")
        
        if seg_metrics:
            print(f"✓ Mean Dice Score: {seg_metrics['mean_dice']:.4f}")
            print(f"✓ Mean IoU Score: {seg_metrics['mean_iou']:.4f}")
        
        print(f"✓ Visualizations saved: confusion_matrix.png, prediction_analysis.png, sample_predictions.png")
        
        return {
            'prediction_analysis': pred_analysis,
            'classification_metrics': class_metrics,
            'segmentation_metrics': seg_metrics
        }


def main():
    """Main function to run comprehensive evaluation"""
    tester = ComprehensiveUNetTester()
    
    try:
        results = tester.run_comprehensive_evaluation()
        print("\n🎉 Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()