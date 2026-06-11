"""
Example usage of the Integrated Gradients U-Net XAI Analysis
"""

import os
import sys
import tensorflow as tf

# Add current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try importing with error handling
try:
    from integrated_gradients_unet import (
        run_integrated_gradients_analysis, 
        IntegratedGradientsExplainer,
        Config
    )
    print("✓ Successfully imported integrated_gradients_unet module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure integrated_gradients_unet.py is in the same directory")
    sys.exit(1)

def load_unet_model():
    """
    Load your trained U-Net model here
    Replace this with your actual model loading code
    """
    # Example - replace with your actual model path
    model_path = "workspace/output_latest/models/unet_model.h5"
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'dice_coefficient': lambda y_true, y_pred: tf.constant(0.0),
                'dice_loss': lambda y_true, y_pred: tf.constant(0.0),
                'iou': lambda y_true, y_pred: tf.constant(0.0),
                'focal_tversky_loss': lambda y_true, y_pred: tf.constant(0.0)
            }
        )
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your model is saved and the path is correct")
        return None


def main():
    """Main function to run integrated gradients analysis"""
    
    # Create output directories
    Config.create_directories()
    
    # Load the trained U-Net model
    print("Loading U-Net model...")
    unet_model = load_unet_model()
    
    if unet_model is None:
        print("Cannot proceed without a trained model")
        return
    
    # Run analysis on different samples
    sample_indices = [0, 5, 10, 15, 20]  # Analyze multiple samples
    
    for idx in sample_indices:
        print(f"\n{'='*60}")
        print(f"ANALYZING SAMPLE {idx}")
        print(f"{'='*60}")
        
        try:
            # Run integrated gradients analysis
            results = run_integrated_gradients_analysis(
                unet_model=unet_model,
                image_index=idx,
                steps=50,  # Number of integration steps
                target_mode='mean_confidence'  # or 'max_confidence', 'total_area'
            )
            
            print(f"Analysis completed for sample {idx}")
            print(f"Patient ID: {results['sample_info']['patient_id']}")
            print(f"Classification: {results['sample_info']['class_name']}")
            
        except Exception as e:
            print(f"Error analyzing sample {idx}: {e}")
            continue
    
    print("\nAll analyses completed!")


def analyze_single_sample(model, image_index=0):
    """
    Analyze a single sample with custom parameters
    """
    print(f"Analyzing single sample at index {image_index}")
    
    # Initialize the explainer with custom parameters
    explainer = IntegratedGradientsExplainer(model, steps=100)
    
    # Run the full analysis
    results = run_integrated_gradients_analysis(
        unet_model=model,
        image_index=image_index,
        steps=100,
        target_mode='mean_confidence'
    )
    
    return results


def batch_analysis_example(model, start_idx=0, num_samples=5):
    """
    Example of analyzing multiple samples in batch
    """
    print(f"Running batch analysis on {num_samples} samples starting from index {start_idx}")
    
    results_list = []
    
    for i in range(start_idx, start_idx + num_samples):
        try:
            print(f"\nProcessing sample {i}...")
            results = run_integrated_gradients_analysis(
                unet_model=model,
                image_index=i,
                steps=30,  # Fewer steps for faster batch processing
                target_mode='mean_confidence'
            )
            results_list.append(results)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"\nBatch analysis completed. Processed {len(results_list)} samples successfully.")
    return results_list


if __name__ == "__main__":
    # Run the main analysis
    main()
    
    # Uncomment below for custom analysis examples:
    
    # # Load model for custom analysis
    # model = load_unet_model()
    # if model:
    #     # Single sample analysis
    #     single_result = analyze_single_sample(model, image_index=3)
    #     
    #     # Batch analysis
    #     batch_results = batch_analysis_example(model, start_idx=0, num_samples=3)