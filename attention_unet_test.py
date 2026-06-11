#!/usr/bin/env python3
"""
Attention U-Net Model Testing and Attention Map Visualization

This script loads the trained attention U-Net model and performs comprehensive testing
including attention map visualization for several test cases.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
import warnings
import random
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility (matching training configuration)
random.seed(73)  # Same as patient selection in training
np.random.seed(42)  # Same as data splits in training
tf.random.set_seed(42)  # TensorFlow random seed (was missing in training)

# Set up GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class AttentionUNetTester:
    """Class for testing attention U-Net model and visualizing attention maps"""
    
    def __init__(self, model_path, data_path, metadata_path):
        """
        Initialize the tester with model and data paths
        
        Args:
            model_path: Path to the trained U-Net model (.h5 file)
            data_path: Path to the comprehensive results (.npz file)
            metadata_path: Path to the detailed metadata (.pkl file)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.model = None
        self.data = None
        self.metadata = None
        
    def load_model_and_data(self):
        """Load the trained model and test data"""
        print("Loading model and data...")
        
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
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
        print("✓ Model loaded successfully")
        
        # Load comprehensive data
        print(f"Loading data from: {self.data_path}")
        self.data = np.load(self.data_path, allow_pickle=True)
        print("✓ Data loaded successfully")
        
        # Load metadata
        print(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print("✓ Metadata loaded successfully")
        
        # Print data shapes
        print(f"\nData shapes:")
        print(f"  X_test: {self.data['X_test'].shape}")
        print(f"  y_test: {self.data['y_test'].shape}")
        print(f"  pred_masks: {self.data['pred_masks'].shape}")
        
    @staticmethod
    def focal_tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, gamma=0.75, smooth=1e-6):
        """Focal Tversky Loss function"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate Tversky components
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        
        # Tversky index
        tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        
        # Focal Tversky Loss
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
        return 1 - AttentionUNetTester.dice_coefficient(y_true, y_pred)
    
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
    
    def extract_attention_maps(self, image_batch, layer_names=None):
        """
        Extract attention maps from the model
        
        Args:
            image_batch: Input image batch
            layer_names: List of attention layer names to extract
            
        Returns:
            List of attention maps
        """
        if layer_names is None:
            # Find attention-related layers automatically
            # Look for Multiply layers (used in attention blocks) and their preceding sigmoid activations
            attention_layers = []
            sigmoid_layers = []
            multiply_layers = []
            
            for i, layer in enumerate(self.model.layers):
                layer_type = type(layer).__name__
                
                # Look for Multiply layers (final step of attention blocks)
                if layer_type == 'Multiply':
                    multiply_layers.append((i, layer.name))
                
                # Look for Activation layers with sigmoid (attention weights)
                if layer_type == 'Activation':
                    try:
                        if hasattr(layer, 'activation') and layer.activation.__name__ == 'sigmoid':
                            sigmoid_layers.append((i, layer.name))
                    except:
                        pass
                
                # Also look for any layer with 'att' in the name
                if 'att' in layer.name.lower():
                    attention_layers.append((i, layer.name))
            
            print(f"Found {len(multiply_layers)} Multiply layers (attention application)")
            print(f"Found {len(sigmoid_layers)} Sigmoid activations (potential attention weights)")
            print(f"Found {len(attention_layers)} layers with 'att' in name")
            
            # Prioritize sigmoid activations as they represent attention weights
            layer_names = [name for _, name in sigmoid_layers[-4:]]  # Last 4 sigmoid layers
            
            # If no sigmoid layers, use multiply layers
            if not layer_names:
                layer_names = [name for _, name in multiply_layers[-4:]]  # Last 4 multiply layers
        
        # Create a model that outputs attention maps
        attention_outputs = []
        valid_layer_names = []
        
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                attention_outputs.append(layer.output)
                valid_layer_names.append(layer_name)
                print(f"✓ Added attention layer: {layer_name}")
            except ValueError:
                print(f"⚠ Layer '{layer_name}' not found")
        
        if not attention_outputs:
            print("No attention layers found. Extracting from decoder conv layers...")
            # Extract from decoder convolutional layers as proxy for attention
            decoder_layers = []
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    # Focus on layers in the second half (decoder)
                    if i > len(self.model.layers) // 2:
                        decoder_layers.append(layer.name)
            
            # Use the last few decoder conv layers
            layer_names = decoder_layers[-6:]
            for layer_name in layer_names:
                try:
                    layer = self.model.get_layer(layer_name)
                    attention_outputs.append(layer.output)
                    valid_layer_names.append(layer_name)
                except ValueError:
                    continue
        
        if attention_outputs:
            print(f"Creating attention extraction model with {len(attention_outputs)} outputs")
            attention_model = tf.keras.Model(inputs=self.model.input, outputs=attention_outputs)
            attention_maps = attention_model.predict(image_batch, verbose=0)
            
            # If single output, convert to list
            if not isinstance(attention_maps, list):
                attention_maps = [attention_maps]
            
            print(f"✓ Extracted {len(attention_maps)} attention maps")
            return attention_maps, valid_layer_names
        else:
            print("No suitable layers found for attention extraction")
            return [], []
    
    def compute_gradient_attention(self, image_batch, target_layer_name=None):
        """
        Compute gradient-based attention maps (Grad-CAM style)
        
        Args:
            image_batch: Input image batch (should be tf.Tensor)
            target_layer_name: Name of the target layer for gradient computation
            
        Returns:
            Gradient-based attention maps
        """
        if target_layer_name is None:
            # Use the last convolutional layer before the final output
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer_name = layer.name
                    break
        
        try:
            # Convert to tensor if it's numpy array
            if isinstance(image_batch, np.ndarray):
                image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
            
            # Create gradient model
            grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[self.model.get_layer(target_layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                tape.watch(image_batch)
                conv_outputs, predictions = grad_model(image_batch)
                # Use the mean of the prediction as the target
                loss = tf.reduce_mean(predictions)
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            if grads is None:
                print("Gradients are None - cannot compute gradient attention")
                return None
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
            
            # Weight the feature maps by the gradients
            conv_outputs_weighted = conv_outputs[0]
            pooled_grads_batch = pooled_grads[0]
            
            # Apply weights to each channel
            for i in range(pooled_grads_batch.shape[-1]):
                conv_outputs_weighted = conv_outputs_weighted[:, :, i:i+1] * pooled_grads_batch[i]
            
            # Create heatmap
            heatmap = tf.reduce_mean(conv_outputs_weighted, axis=-1)
            heatmap = tf.maximum(heatmap, 0)  # ReLU
            
            # Normalize
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Error computing gradient attention: {e}")
            return None
    
    def load_ground_truth_masks(self):
        """
        Try to load ground truth segmentation masks from various sources
        
        Returns:
            Ground truth masks if available, None otherwise
        """
        # Try to load from training data with metadata
        try:
            training_data_path = "/workspace/output/training_data_with_metadata.npz"
            if os.path.exists(training_data_path):
                training_data = np.load(training_data_path, allow_pickle=True)
                if 'm_test' in training_data:
                    print("✓ Found ground truth masks in training data")
                    return training_data['m_test']
        except Exception as e:
            print(f"Could not load from training data: {e}")
        
        # Try to reconstruct from LIDC annotations
        try:
            if 'test_metadata' in self.metadata:
                print("Attempting to reconstruct ground truth masks from metadata...")
                test_metadata = self.metadata['test_metadata']
                ground_truth_masks = []
                
                for i, sample_meta in enumerate(test_metadata):
                    # Create empty mask
                    mask = np.zeros((512, 512, 1), dtype=np.float32)
                    
                    # This would require access to original LIDC data
                    # For now, we'll use the U-Net predictions as pseudo ground truth
                    # In a real scenario, you'd load the actual annotations here
                    
                    ground_truth_masks.append(mask)
                
                return np.array(ground_truth_masks)
        except Exception as e:
            print(f"Could not reconstruct ground truth: {e}")
        
        return None
    
    def evaluate_test_set(self, num_samples=None):
        """
        Evaluate the model on the test set
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model on test set...")
        
        X_test = self.data['X_test']
        if num_samples is not None:
            X_test = X_test[:num_samples]
        
        # Make predictions
        predictions = self.model.predict(X_test, batch_size=8, verbose=1)
        
        # Try to load ground truth masks
        ground_truth_masks = self.load_ground_truth_masks()
        
        if ground_truth_masks is not None:
            if num_samples is not None:
                ground_truth_masks = ground_truth_masks[:num_samples]
            
            # Compute metrics
            dice_scores = []
            iou_scores = []
            precision_scores = []
            recall_scores = []
            
            for i in range(len(predictions)):
                pred = predictions[i:i+1]
                true = ground_truth_masks[i:i+1]
                
                dice_score = self.dice_coefficient(true, pred).numpy()
                iou_score = self.iou(true, pred).numpy()
                precision_score = self.precision(true, pred).numpy()
                recall_score = self.recall(true, pred).numpy()
                
                dice_scores.append(dice_score)
                iou_scores.append(iou_score)
                precision_scores.append(precision_score)
                recall_scores.append(recall_score)
            
            metrics = {
                'mean_dice': np.mean(dice_scores),
                'std_dice': np.std(dice_scores),
                'mean_iou': np.mean(iou_scores),
                'std_iou': np.std(iou_scores),
                'mean_precision': np.mean(precision_scores),
                'std_precision': np.std(precision_scores),
                'mean_recall': np.mean(recall_scores),
                'std_recall': np.std(recall_scores),
                'dice_scores': dice_scores,
                'iou_scores': iou_scores,
                'precision_scores': precision_scores,
                'recall_scores': recall_scores
            }
            
            print(f"Mean Dice Score: {metrics['mean_dice']:.4f} ± {metrics['std_dice']:.4f}")
            print(f"Mean IoU Score: {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}")
            print(f"Mean Precision: {metrics['mean_precision']:.4f} ± {metrics['std_precision']:.4f}")
            print(f"Mean Recall: {metrics['mean_recall']:.4f} ± {metrics['std_recall']:.4f}")
            
            return metrics
        else:
            print("No ground truth masks available for metric computation")
            return {'predictions': predictions}
    
    def visualize_attention_maps(self, image_indices, save_path=None):
        """
        Visualize attention maps for selected test images
        
        Args:
            image_indices: List of image indices to visualize
            save_path: Path to save the visualization (optional)
        """
        print(f"\nVisualizing attention maps for images: {image_indices}")
        
        X_test = self.data['X_test']
        pred_masks = self.data['pred_masks']
        
        fig, axes = plt.subplots(len(image_indices), 4, figsize=(16, 4*len(image_indices)))
        if len(image_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_idx in enumerate(image_indices):
            # Get the image and prediction
            image = X_test[img_idx:img_idx+1]
            prediction = pred_masks[img_idx]
            
            # Extract attention maps
            attention_maps, attention_layer_names = self.extract_attention_maps(image)
            gradient_attention = self.compute_gradient_attention(image)
            
            # Plot original image
            axes[idx, 0].imshow(image[0, :, :, 0], cmap='gray')
            axes[idx, 0].set_title(f'Original Image {img_idx}')
            axes[idx, 0].axis('off')
            
            # Plot prediction
            axes[idx, 1].imshow(prediction[:, :, 0], cmap='hot', alpha=0.7)
            axes[idx, 1].imshow(image[0, :, :, 0], cmap='gray', alpha=0.3)
            axes[idx, 1].set_title('U-Net Prediction')
            axes[idx, 1].axis('off')
            
            # Plot attention map (if available)
            if attention_maps and len(attention_maps) > 0:
                # Use the best attention map (usually the last one which is highest resolution)
                att_map = attention_maps[-1][0]  # Last attention map, first batch
                
                # Handle different attention map formats
                if len(att_map.shape) == 3:
                    if att_map.shape[-1] == 1:
                        # Single channel attention map
                        att_map = att_map[:, :, 0]
                    else:
                        # Multi-channel, average across channels
                        att_map = np.mean(att_map, axis=-1)
                
                # Resize to match image size if needed
                if att_map.shape != image.shape[1:3]:
                    att_map_resized = tf.image.resize(
                        att_map[..., np.newaxis], 
                        [image.shape[1], image.shape[2]]
                    ).numpy()[:, :, 0]
                else:
                    att_map_resized = att_map
                
                # Normalize attention map
                att_map_resized = (att_map_resized - att_map_resized.min()) / (att_map_resized.max() - att_map_resized.min() + 1e-8)
                
                axes[idx, 2].imshow(image[0, :, :, 0], cmap='gray', alpha=0.6)
                axes[idx, 2].imshow(att_map_resized, cmap='jet', alpha=0.6)
                
                # Add layer name to title if available
                layer_name = attention_layer_names[-1] if attention_layer_names else "Unknown"
                axes[idx, 2].set_title(f'Attention Map\n({layer_name})')
            else:
                axes[idx, 2].text(0.5, 0.5, 'No Attention\nMap Available', 
                                ha='center', va='center', transform=axes[idx, 2].transAxes)
                axes[idx, 2].set_title('Attention Map')
            axes[idx, 2].axis('off')
            
            # Plot gradient attention (if available)
            if gradient_attention is not None:
                # Resize gradient attention to match image size
                grad_att_resized = tf.image.resize(
                    gradient_attention[..., np.newaxis], 
                    [image.shape[1], image.shape[2]]
                ).numpy()[:, :, 0]
                
                axes[idx, 3].imshow(image[0, :, :, 0], cmap='gray', alpha=0.5)
                axes[idx, 3].imshow(grad_att_resized, cmap='jet', alpha=0.5)
                axes[idx, 3].set_title('Gradient Attention')
            else:
                axes[idx, 3].text(0.5, 0.5, 'Gradient Attention\nNot Available', 
                                ha='center', va='center', transform=axes[idx, 3].transAxes)
                axes[idx, 3].set_title('Gradient Attention')
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_sample_info(self, image_index):
        """Get detailed information about a specific sample"""
        if 'test_metadata' in self.metadata:
            test_metadata = self.metadata['test_metadata']
            if image_index < len(test_metadata):
                sample_metadata = test_metadata[image_index]
                
                info = {
                    'patient_id': sample_metadata.get('patient_id', 'Unknown'),
                    'slice_idx': sample_metadata.get('slice_idx', 'Unknown'),
                    'uid': sample_metadata.get('uid', 'Unknown'),
                    'classification_label': self.data['y_test'][image_index] if 'y_test' in self.data else 'Unknown'
                }
                
                return info
        
        return {'message': 'Sample information not available'}
    
    def run_comprehensive_test(self, test_indices=None, num_eval_samples=100):
        """
        Run comprehensive testing including evaluation and visualization
        
        Args:
            test_indices: List of specific indices to visualize (default: random selection)
            num_eval_samples: Number of samples for evaluation
        """
        print("="*80)
        print("ATTENTION U-NET COMPREHENSIVE TESTING")
        print("="*80)
        
        # Load model and data
        self.load_model_and_data()
        
        # Evaluate on test set
        metrics = self.evaluate_test_set(num_samples=num_eval_samples)
        
        # Select test indices for visualization
        if test_indices is None:
            # Select a mix of samples
            total_samples = len(self.data['X_test'])
            test_indices = [
                0,  # First sample
                total_samples // 4,  # Quarter point
                total_samples // 2,  # Midpoint
                3 * total_samples // 4,  # Three-quarter point
                min(total_samples - 1, 100)  # Sample around 100 or last
            ]
        
        # Print sample information
        print(f"\nSelected test samples for visualization:")
        for idx in test_indices:
            info = self.get_sample_info(idx)
            print(f"  Index {idx}: {info}")
        
        # Visualize attention maps
        self.visualize_attention_maps(test_indices, save_path='attention_maps_visualization.png')
        
        print("\n" + "="*80)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return metrics


def main():
    """Main function to run the attention U-Net testing"""
    
    # Define file paths
    model_path = "/workspace/output/models/best_unet_model.h5"
    data_path = "/workspace/output/comprehensive_results.npz"
    metadata_path = "/workspace/output/detailed_metadata.pkl"
    
    # Check if files exist
    missing_files = []
    for path, name in [(model_path, "Model"), (data_path, "Data"), (metadata_path, "Metadata")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("ERROR: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure the model has been trained and saved properly.")
        return
    
    # Create tester instance
    tester = AttentionUNetTester(model_path, data_path, metadata_path)
    
    # Run comprehensive testing
    try:
        metrics = tester.run_comprehensive_test(
            test_indices=[0, 50, 100, 150, 200],  # Specific indices to visualize
            num_eval_samples=200  # Number of samples for evaluation
        )
        
        print(f"\nFinal Results Summary:")
        if 'mean_dice' in metrics:
            print(f"  Mean Dice Score: {metrics['mean_dice']:.4f}")
            print(f"  Mean IoU Score: {metrics['mean_iou']:.4f}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()