#!/usr/bin/env python3
"""
Specialized Attention Visualization for U-Net

This script specifically targets the attention mechanism in the U-Net model
by identifying Multiply and Sigmoid layers that implement the attention blocks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

class AttentionVisualizer:
    """Specialized class for visualizing attention mechanisms in U-Net"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.attention_layers = []
        
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
        model_path = "/workspace/output/models/best_unet_model.h5"
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✓ Model loaded successfully")
        
        # Load comprehensive data
        data_path = "/workspace/output/comprehensive_results.npz"
        print(f"Loading data from: {data_path}")
        self.data = np.load(data_path, allow_pickle=True)
        print("✓ Data loaded successfully")
        
        # Analyze model architecture for attention layers
        self.analyze_attention_architecture()
    
    def analyze_attention_architecture(self):
        """Analyze the model architecture to find attention-related layers"""
        print("\n" + "="*60)
        print("ANALYZING ATTENTION ARCHITECTURE")
        print("="*60)
        
        multiply_layers = []
        sigmoid_activations = []
        conv_before_sigmoid = []
        
        for i, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            
            # Find Multiply layers (final attention application)
            if layer_type == 'Multiply':
                multiply_layers.append((i, layer.name, layer.output_shape))
                print(f"Multiply layer {i}: {layer.name} | {layer.output_shape}")
            
            # Find Sigmoid activations (attention weights)
            elif layer_type == 'Activation':
                try:
                    if hasattr(layer, 'activation') and layer.activation.__name__ == 'sigmoid':
                        sigmoid_activations.append((i, layer.name, layer.output_shape))
                        print(f"Sigmoid activation {i}: {layer.name} | {layer.output_shape}")
                        
                        # Look for the Conv2D layer just before this sigmoid
                        if i > 0:
                            prev_layer = self.model.layers[i-1]
                            if isinstance(prev_layer, tf.keras.layers.Conv2D):
                                conv_before_sigmoid.append((i-1, prev_layer.name, prev_layer.output_shape))
                except:
                    pass
        
        print(f"\nFound {len(multiply_layers)} Multiply layers")
        print(f"Found {len(sigmoid_activations)} Sigmoid activations")
        print(f"Found {len(conv_before_sigmoid)} Conv2D layers before sigmoid")
        
        # Store attention layers for extraction
        self.attention_layers = {
            'multiply': multiply_layers,
            'sigmoid': sigmoid_activations,
            'conv_before_sigmoid': conv_before_sigmoid
        }
        
        return self.attention_layers
    
    def extract_attention_weights(self, image_batch):
        """Extract attention weights from sigmoid activations"""
        print("\nExtracting attention weights...")
        
        # Get sigmoid activation layers (these are the attention weights)
        sigmoid_layers = self.attention_layers['sigmoid']
        
        if not sigmoid_layers:
            print("No sigmoid layers found for attention extraction")
            return None
        
        # Create model to extract sigmoid outputs
        sigmoid_outputs = []
        layer_names = []
        
        for idx, layer_name, output_shape in sigmoid_layers:
            try:
                layer = self.model.get_layer(layer_name)
                sigmoid_outputs.append(layer.output)
                layer_names.append(layer_name)
                print(f"✓ Added sigmoid layer: {layer_name} | {output_shape}")
            except ValueError:
                print(f"⚠ Could not find layer: {layer_name}")
        
        if sigmoid_outputs:
            attention_model = tf.keras.Model(inputs=self.model.input, outputs=sigmoid_outputs)
            attention_weights = attention_model.predict(image_batch, verbose=0)
            
            if not isinstance(attention_weights, list):
                attention_weights = [attention_weights]
            
            print(f"✓ Extracted {len(attention_weights)} attention weight maps")
            return attention_weights, layer_names
        
        return None, []
    
    def visualize_attention_progression(self, image_index, save_path=None):
        """Visualize how attention evolves through the decoder"""
        print(f"\nVisualizing attention progression for image {image_index}")
        
        # Get the image
        image = self.data['X_test'][image_index:image_index+1]
        prediction = self.data['pred_masks'][image_index]
        
        # Extract attention weights
        attention_weights, layer_names = self.extract_attention_weights(image)
        
        if not attention_weights:
            print("No attention weights extracted")
            return
        
        # Create visualization
        num_attentions = len(attention_weights)
        fig, axes = plt.subplots(2, max(3, num_attentions), figsize=(4*max(3, num_attentions), 8))
        
        # First row: Original, Prediction, and attention weights
        axes[0, 0].imshow(image[0, :, :, 0], cmap='gray')
        axes[0, 0].set_title(f'Original Image {image_index}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(prediction[:, :, 0], cmap='hot')
        axes[0, 1].set_title('U-Net Prediction')
        axes[0, 1].axis('off')
        
        # Show attention weights
        for i, (att_weights, layer_name) in enumerate(zip(attention_weights, layer_names)):
            if i < num_attentions and i + 2 < axes.shape[1]:
                # Get attention map for first batch
                att_map = att_weights[0]
                
                # Handle different shapes
                if len(att_map.shape) == 3:
                    if att_map.shape[-1] == 1:
                        att_map = att_map[:, :, 0]
                    else:
                        att_map = np.mean(att_map, axis=-1)
                
                # Resize to match image size
                if att_map.shape != image.shape[1:3]:
                    att_map_resized = tf.image.resize(
                        att_map[..., np.newaxis], 
                        [image.shape[1], image.shape[2]]
                    ).numpy()[:, :, 0]
                else:
                    att_map_resized = att_map
                
                axes[0, i + 2].imshow(att_map_resized, cmap='jet')
                axes[0, i + 2].set_title(f'Attention {i+1}\n{layer_name}')
                axes[0, i + 2].axis('off')
        
        # Second row: Overlays of attention on original image
        axes[1, 0].imshow(image[0, :, :, 0], cmap='gray')
        axes[1, 0].set_title('Original')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(image[0, :, :, 0], cmap='gray', alpha=0.7)
        axes[1, 1].imshow(prediction[:, :, 0], cmap='hot', alpha=0.5)
        axes[1, 1].set_title('Prediction Overlay')
        axes[1, 1].axis('off')
        
        # Overlay attention maps
        for i, (att_weights, layer_name) in enumerate(zip(attention_weights, layer_names)):
            if i < num_attentions and i + 2 < axes.shape[1]:
                att_map = att_weights[0]
                
                if len(att_map.shape) == 3:
                    if att_map.shape[-1] == 1:
                        att_map = att_map[:, :, 0]
                    else:
                        att_map = np.mean(att_map, axis=-1)
                
                if att_map.shape != image.shape[1:3]:
                    att_map_resized = tf.image.resize(
                        att_map[..., np.newaxis], 
                        [image.shape[1], image.shape[2]]
                    ).numpy()[:, :, 0]
                else:
                    att_map_resized = att_map
                
                # Normalize attention map
                att_map_resized = (att_map_resized - att_map_resized.min()) / (att_map_resized.max() - att_map_resized.min() + 1e-8)
                
                axes[1, i + 2].imshow(image[0, :, :, 0], cmap='gray', alpha=0.6)
                axes[1, i + 2].imshow(att_map_resized, cmap='jet', alpha=0.6)
                axes[1, i + 2].set_title(f'Attention Overlay {i+1}')
                axes[1, i + 2].axis('off')
        
        # Hide unused subplots
        for i in range(num_attentions + 2, axes.shape[1]):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def compare_multiple_samples(self, image_indices, save_path=None):
        """Compare attention maps across multiple samples"""
        print(f"\nComparing attention across {len(image_indices)} samples")
        
        fig, axes = plt.subplots(len(image_indices), 4, figsize=(16, 4*len(image_indices)))
        if len(image_indices) == 1:
            axes = axes.reshape(1, -1)
        
        for row, img_idx in enumerate(image_indices):
            # Get image and prediction
            image = self.data['X_test'][img_idx:img_idx+1]
            prediction = self.data['pred_masks'][img_idx]
            
            # Extract attention
            attention_weights, layer_names = self.extract_attention_weights(image)
            
            # Original image
            axes[row, 0].imshow(image[0, :, :, 0], cmap='gray')
            axes[row, 0].set_title(f'Image {img_idx}')
            axes[row, 0].axis('off')
            
            # Prediction
            axes[row, 1].imshow(prediction[:, :, 0], cmap='hot')
            axes[row, 1].set_title('Prediction')
            axes[row, 1].axis('off')
            
            # Best attention map
            if attention_weights and len(attention_weights) > 0:
                # Use the highest resolution attention map
                best_att = attention_weights[-1][0]  # Last attention map, first batch
                
                if len(best_att.shape) == 3:
                    if best_att.shape[-1] == 1:
                        best_att = best_att[:, :, 0]
                    else:
                        best_att = np.mean(best_att, axis=-1)
                
                # Resize if needed
                if best_att.shape != image.shape[1:3]:
                    best_att = tf.image.resize(
                        best_att[..., np.newaxis], 
                        [image.shape[1], image.shape[2]]
                    ).numpy()[:, :, 0]
                
                axes[row, 2].imshow(best_att, cmap='jet')
                axes[row, 2].set_title('Attention Map')
                axes[row, 2].axis('off')
                
                # Overlay
                best_att_norm = (best_att - best_att.min()) / (best_att.max() - best_att.min() + 1e-8)
                axes[row, 3].imshow(image[0, :, :, 0], cmap='gray', alpha=0.6)
                axes[row, 3].imshow(best_att_norm, cmap='jet', alpha=0.6)
                axes[row, 3].set_title('Attention Overlay')
                axes[row, 3].axis('off')
            else:
                axes[row, 2].text(0.5, 0.5, 'No Attention\nAvailable', ha='center', va='center')
                axes[row, 3].text(0.5, 0.5, 'No Attention\nAvailable', ha='center', va='center')
                axes[row, 2].axis('off')
                axes[row, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        plt.show()
    
    # Define the required loss functions and metrics
    @staticmethod
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
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
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
        return 1 - AttentionVisualizer.dice_coefficient(y_true, y_pred)
    
    @staticmethod
    def iou(y_true, y_pred, smooth=1e-6):
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
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        actual_pos = tf.reduce_sum(y_true_flat)
        
        recall_score = (true_pos + smooth) / (actual_pos + smooth)
        return recall_score


def main():
    """Main function to run attention visualization"""
    print("="*80)
    print("ATTENTION U-NET VISUALIZATION")
    print("="*80)
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Load model and data
    visualizer.load_model_and_data()
    
    # Visualize attention progression for a specific image
    print("\n" + "="*60)
    print("VISUALIZING ATTENTION PROGRESSION")
    print("="*60)
    
    # Choose an interesting sample (one with nodules)
    sample_indices = [0, 50, 100, 150, 200]
    
    for idx in sample_indices[:2]:  # Visualize first 2 samples in detail
        visualizer.visualize_attention_progression(
            idx, 
            save_path=f'attention_progression_{idx}.png'
        )
    
    # Compare attention across multiple samples
    print("\n" + "="*60)
    print("COMPARING ATTENTION ACROSS SAMPLES")
    print("="*60)
    
    visualizer.compare_multiple_samples(
        sample_indices, 
        save_path='attention_comparison.png'
    )
    
    print("\n🎉 Attention visualization completed!")
    print("Generated files:")
    print("  - attention_progression_0.png")
    print("  - attention_progression_50.png") 
    print("  - attention_comparison.png")


if __name__ == "__main__":
    main()