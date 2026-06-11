#!/usr/bin/env python3
"""
Inspect the actual layer names and architecture of the trained U-Net model
"""

import tensorflow as tf
import numpy as np

# Define custom objects for model loading
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

def main():
    # Load the model
    custom_objects = {
        'focal_tversky_loss': focal_tversky_loss,
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }
    
    model_path = "/workspace/output/models/best_unet_model.h5"
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*80)
    
    print(f"\nModel Summary:")
    model.summary()
    
    print(f"\n" + "="*80)
    print("ALL LAYER NAMES AND TYPES")
    print("="*80)
    
    attention_candidates = []
    conv_layers = []
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        print(f"{i:3d}: {layer.name:25s} | {layer_type:20s} | Output: {layer.output_shape}")
        
        # Look for potential attention layers
        if any(keyword in layer.name.lower() for keyword in ['att', 'attention', 'gate']):
            attention_candidates.append((i, layer.name, layer_type))
        
        # Collect conv layers for potential attention extraction
        if 'Conv2D' in layer_type:
            conv_layers.append((i, layer.name, layer_type, layer.output_shape))
    
    print(f"\n" + "="*80)
    print("POTENTIAL ATTENTION LAYERS")
    print("="*80)
    
    if attention_candidates:
        print("Found potential attention layers:")
        for idx, name, layer_type in attention_candidates:
            print(f"  {idx}: {name} ({layer_type})")
    else:
        print("No explicit attention layers found in layer names.")
        print("This suggests the attention mechanism is implemented as part of other layers.")
    
    print(f"\n" + "="*80)
    print("CONVOLUTIONAL LAYERS (for attention extraction)")
    print("="*80)
    
    print("Key convolutional layers that could be used for attention visualization:")
    # Focus on layers that are likely to be attention-related based on the U-Net architecture
    decoder_layers = []
    for idx, name, layer_type, output_shape in conv_layers:
        if any(keyword in name for keyword in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']):
            print(f"  Encoder: {idx}: {name} | {output_shape}")
        elif idx > len(conv_layers) // 2:  # Likely decoder layers
            decoder_layers.append((idx, name, layer_type, output_shape))
    
    print("\nDecoder layers (good for attention visualization):")
    for idx, name, layer_type, output_shape in decoder_layers[-8:]:  # Last 8 decoder layers
        print(f"  Decoder: {idx}: {name} | {output_shape}")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS FOR ATTENTION EXTRACTION")
    print("="*80)
    
    print("Based on the U-Net architecture from your notebook, the attention mechanism")
    print("is likely implemented within the decoder blocks. Try these layers for attention:")
    
    # Suggest specific layers based on typical U-Net attention architecture
    suggested_layers = []
    for idx, name, layer_type, output_shape in conv_layers:
        # Look for layers that might be attention outputs
        if any(pattern in name for pattern in ['conv2d_', 'activation_']):
            if len(output_shape) == 4 and output_shape[-1] in [32, 64, 128, 256]:  # Typical attention channels
                suggested_layers.append((idx, name, output_shape))
    
    print("\nSuggested layers for attention visualization:")
    for idx, name, output_shape in suggested_layers[-10:]:  # Last 10 relevant layers
        print(f"  {idx}: {name} | {output_shape}")

if __name__ == "__main__":
    main()