"""
Standalone Integrated Gradients Analysis for U-Net XAI
Complete implementation for integrated gradients explanation of U-Net segmentation models
Includes batch processing capabilities and comprehensive visualization tools
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

import pydicom
import xml.etree.ElementTree as ET
from skimage.draw import polygon
from tqdm import tqdm
import tensorflow as tf

print("Standalone Integrated Gradients - All imports completed")


class Config:
    """Configuration parameters for integrated gradients analysis"""
    DATA_DIR = "/workspace/storage/lidc_dataset"
    OUTPUT_DIR = "workspace/output"
    
    IMAGE_SIZE = (512, 512)
    HU_MIN = -1000
    HU_MAX = 400
    
    SEGMENTATION_THRESHOLD = 0.4
    IG_STEPS = 50
    
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


def load_saved_data():
    """Load preprocessed training data and metadata from disk"""
    print("Loading saved data files...")
    
    comprehensive_path = os.path.join(Config.OUTPUT_DIR, 'comprehensive_results.npz')
    comprehensive_data = np.load(comprehensive_path)
    
    metadata_path = os.path.join(Config.OUTPUT_DIR, 'detailed_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        detailed_metadata = pickle.load(f)
    
    print(f"Loaded data - X_test shape: {comprehensive_data['X_test'].shape}")
    print(f"Loaded data - y_test shape: {comprehensive_data['y_test'].shape}")
    
    return comprehensive_data, detailed_metadata


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer for U-Net segmentation models
    
    Implements integrated gradients attribution using Riemann sum approximation
    to explain model predictions by computing the path integral of gradients.
    """
    
    def __init__(self, model, steps=50):
        """
        Initialize the explainer
        
        Args:
            model: Trained U-Net model
            steps: Number of steps for Riemann sum approximation
        """
        self.model = model
        self.steps = steps
    
    def explain(self, input_image, baseline=None, target_mode='mean_confidence'):
        """
        Generate integrated gradients explanation
        
        Computes attributions by integrating gradients along the path from
        baseline to input image: IG = (x - x') * integral(gradients)
        
        Args:
            input_image: Input tensor (batch_size=1, height, width, channels)
            baseline: Baseline image (default: zeros)
            target_mode: How to compute scalar target from segmentation output
                        Options: 'mean_confidence', 'max_confidence', 'total_area'
        
        Returns:
            integrated_grads: Attribution values for each input pixel
            prediction: Model prediction for the input
        """
        if not tf.is_tensor(input_image):
            input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        
        if baseline is None:
            baseline = tf.zeros_like(input_image)
        elif not tf.is_tensor(baseline):
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
        
        # Generate interpolation path from baseline to input
        alphas = tf.linspace(0.0, 1.0, self.steps + 1)
        path_gradients = []
        
        for alpha in tqdm(alphas, desc="Computing integrated gradients"):
            # Interpolate: baseline + alpha * (input - baseline)
            interpolated = baseline + alpha * (input_image - baseline)
            
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                prediction = self.model(interpolated)
                
                # Convert segmentation output to scalar target
                if target_mode == 'mean_confidence':
                    target = tf.reduce_mean(prediction)
                elif target_mode == 'max_confidence':
                    target = tf.reduce_max(prediction)
                elif target_mode == 'total_area':
                    target = tf.reduce_sum(prediction)
                else:
                    target = tf.reduce_mean(prediction)
            
            gradient = tape.gradient(target, interpolated)
            path_gradients.append(gradient)
        
        # Approximate integral using average of gradients (Riemann sum)
        integrated_grads = tf.reduce_mean(tf.stack(path_gradients), axis=0)
        
        # Scale by input difference: (x - x') * integrated_gradients
        integrated_grads = integrated_grads * (input_image - baseline)
        
        final_prediction = self.model(input_image)
        
        return integrated_grads, final_prediction


def visualize_integrated_gradients(image, prediction, integrated_grads, sample_info, save_path=None):
    """
    Create comprehensive visualization of integrated gradients analysis
    
    Displays original image, prediction, overlay, and attribution maps with statistics
    """
    image_2d = np.squeeze(image)
    pred_2d = np.squeeze(prediction)
    ig_2d = np.squeeze(integrated_grads)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data row
    axes[0, 0].imshow(image_2d, cmap='gray')
    axes[0, 0].set_title('Original CT Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_2d, cmap='hot')
    axes[0, 1].set_title('U-Net Prediction')
    axes[0, 1].axis('off')
    
    pred_binary = (pred_2d > Config.SEGMENTATION_THRESHOLD).astype(np.uint8)
    overlay = np.zeros((*image_2d.shape, 3))
    overlay[:, :, 0] = image_2d
    overlay[:, :, 1] = pred_binary
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Image + Prediction')
    axes[0, 2].axis('off')
    
    # Attribution analysis row
    axes[1, 0].imshow(ig_2d, cmap='RdBu_r')
    axes[1, 0].set_title('Integrated Gradients')
    axes[1, 0].axis('off')
    
    ig_enhanced = ig_2d * 1000
    axes[1, 1].imshow(ig_enhanced, cmap='RdBu_r')
    axes[1, 1].set_title('Enhanced IG (x1000)')
    axes[1, 1].axis('off')
    
    stats_text = f"""
SAMPLE INFO:
Patient ID: {sample_info.get('patient_id', 'N/A')}
Class: {sample_info.get('class_name', 'N/A')}

IG STATISTICS:
Min: {np.min(ig_2d):.2e}
Max: {np.max(ig_2d):.2e}
Mean: {np.mean(ig_2d):.2e}
Std: {np.std(ig_2d):.2e}

PREDICTION:
Max confidence: {np.max(pred_2d):.3f}
Mean confidence: {np.mean(pred_2d):.3f}
Predicted area: {np.sum(pred_binary)} pixels
"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Integrated Gradients Analysis - Patient {sample_info.get("patient_id", "N/A")} '
                f'({sample_info.get("class_name", "Unknown")})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def load_unet_model():
    """
    Load trained U-Net model with custom objects handling
    
    Returns:
        Loaded model or None if loading fails
    """
    model_path = "workspace/output/models/best_unet_model.h5"
    
    # Define placeholder functions for custom metrics and losses
    custom_objects = {
        'dice_coefficient': lambda y_true, y_pred: tf.constant(0.0),
        'dice_loss': lambda y_true, y_pred: tf.constant(0.0),
        'iou': lambda y_true, y_pred: tf.constant(0.0),
        'focal_tversky_loss': lambda y_true, y_pred: tf.constant(0.0)
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model path and ensure the model file exists")
        return None


def run_integrated_gradients_analysis(unet_model, image_index, steps=50, target_mode='mean_confidence'):
    """
    Run integrated gradients analysis for a specific image with visualization
    
    Args:
        unet_model: Trained U-Net model
        image_index: Index of test image to analyze
        steps: Number of integration steps
        target_mode: Method to compute target from segmentation output
    
    Returns:
        Dictionary containing analysis results and visualizations
    """
    print("="*60)
    print("INTEGRATED GRADIENTS U-NET ANALYSIS")
    print("="*60)
    
    comprehensive_data, detailed_metadata = load_saved_data()
    
    test_metadata = detailed_metadata['test_metadata']
    sample_metadata = test_metadata[image_index]
    
    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    class_label = comprehensive_data['y_test'][image_index]
    
    sample_info = {
        'patient_id': sample_metadata['patient_id'],
        'uid': sample_metadata['uid'],
        'slice_idx': sample_metadata['slice_idx'],
        'class_label': class_label,
        'class_name': class_names.get(class_label, 'Unknown'),
        'malignancy_score': sample_metadata.get('malignancy', 0)
    }
    
    print(f"Analyzing Patient {sample_info['patient_id']} - {sample_info['class_name']}")
    
    image_to_explain = comprehensive_data['X_test'][image_index:image_index+1]
    unet_prediction = comprehensive_data['pred_masks'][image_index:image_index+1]
    
    print(f"Data shapes - Image: {image_to_explain.shape}")
    
    explainer = IntegratedGradientsExplainer(unet_model, steps=steps)
    
    print(f"Computing integrated gradients with {steps} steps...")
    integrated_grads, prediction = explainer.explain(
        image_to_explain, 
        target_mode=target_mode
    )
    
    print("Creating visualization...")
    save_path = os.path.join(Config.OUTPUT_DIR, f'ig_analysis_patient_{sample_info["patient_id"]}.png')
    
    visualize_integrated_gradients(
        image=image_to_explain,
        prediction=unet_prediction,
        integrated_grads=integrated_grads.numpy(),
        sample_info=sample_info,
        save_path=save_path
    )
    
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'sample_info': sample_info,
        'integrated_grads': integrated_grads.numpy(),
        'prediction': prediction.numpy(),
        'image': image_to_explain
    }


def run_batch_integrated_gradients(unet_model, start_idx=0, end_idx=None, steps=30, 
                                  target_mode='mean_confidence', save_results=True):
    """
    Run integrated gradients analysis on a batch of test samples
    
    Args:
        unet_model: Trained U-Net model
        start_idx: Starting index in test set
        end_idx: Ending index in test set (None = all remaining)
        steps: Number of integration steps
        target_mode: Target computation mode
        save_results: Whether to save results to files
    
    Returns:
        Tuple of (batch_results, failed_samples)
    """
    print("="*80)
    print("BATCH INTEGRATED GRADIENTS ANALYSIS")
    print("="*80)
    
    comprehensive_data, detailed_metadata = load_saved_data()
    test_metadata = detailed_metadata['test_metadata']
    
    total_samples = len(comprehensive_data['X_test'])
    if end_idx is None:
        end_idx = total_samples
    
    end_idx = min(end_idx, total_samples)
    
    print(f"Processing samples {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
    print(f"Integration steps: {steps}")
    print(f"Target mode: {target_mode}")
    
    batch_results = []
    failed_samples = []
    
    for idx in range(start_idx, end_idx):
        try:
            print(f"\n{'='*40}")
            print(f"PROCESSING SAMPLE {idx} ({idx-start_idx+1}/{end_idx-start_idx})")
            print(f"{'='*40}")
            
            results = run_integrated_gradients_analysis_batch(
                unet_model=unet_model,
                image_index=idx,
                steps=steps,
                target_mode=target_mode
            )
            
            batch_results.append(results)
            
            print(f"Sample {idx} completed - Patient {results['sample_info']['patient_id']}")
            
            if save_results:
                save_individual_results(results, idx)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed_samples.append(idx)
            continue
    
    print(f"\n{'='*80}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(batch_results)} samples")
    print(f"Failed samples: {len(failed_samples)}")
    if failed_samples:
        print(f"Failed indices: {failed_samples}")
    
    if save_results and batch_results:
        save_batch_results(batch_results, start_idx, end_idx, steps)
    
    return batch_results, failed_samples


def run_integrated_gradients_analysis_batch(unet_model, image_index, steps=50, target_mode='mean_confidence'):
    """
    Run integrated gradients analysis without visualization for batch processing
    
    Optimized version that skips visualization to save time and memory
    
    Args:
        unet_model: Trained U-Net model
        image_index: Index of test image to analyze
        steps: Number of integration steps
        target_mode: Target computation mode
    
    Returns:
        Dictionary containing analysis results without visualization
    """
    comprehensive_data, detailed_metadata = load_saved_data()
    
    test_metadata = detailed_metadata['test_metadata']
    sample_metadata = test_metadata[image_index]
    
    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    class_label = comprehensive_data['y_test'][image_index]
    
    sample_info = {
        'patient_id': sample_metadata['patient_id'],
        'uid': sample_metadata['uid'],
        'slice_idx': sample_metadata['slice_idx'],
        'class_label': class_label,
        'class_name': class_names.get(class_label, 'Unknown'),
        'malignancy_score': sample_metadata.get('malignancy', 0)
    }
    
    image_to_explain = comprehensive_data['X_test'][image_index:image_index+1]
    unet_prediction = comprehensive_data['pred_masks'][image_index:image_index+1]
    
    explainer = IntegratedGradientsExplainer(unet_model, steps=steps)
    
    integrated_grads, prediction = explainer.explain(
        image_to_explain, 
        target_mode=target_mode
    )
    
    return {
        'sample_info': sample_info,
        'integrated_grads': integrated_grads.numpy(),
        'prediction': prediction.numpy(),
        'image': image_to_explain
    }


def save_individual_results(results, sample_idx):
    """
    Save individual sample results to disk
    
    Creates a directory for each sample with integrated gradients and metadata
    """
    try:
        sample_dir = os.path.join(Config.OUTPUT_DIR, f'ig_sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)
        
        np.save(os.path.join(sample_dir, 'integrated_gradients.npy'), 
                results['integrated_grads'])
        
        info_path = os.path.join(sample_dir, 'sample_info.txt')
        with open(info_path, 'w') as f:
            for key, value in results['sample_info'].items():
                f.write(f"{key}: {value}\n")
        
    except Exception as e:
        print(f"Failed to save individual results: {e}")


def save_batch_results(batch_results, start_idx, end_idx, steps):
    """Save aggregated batch results and summary statistics"""
    try:
        batch_dir = os.path.join(Config.OUTPUT_DIR, f'ig_batch_results_{start_idx}_{end_idx}')
        os.makedirs(batch_dir, exist_ok=True)
        
        all_igs = []
        all_info = []
        
        for result in batch_results:
            all_igs.append(result['integrated_grads'])
            all_info.append(result['sample_info'])
        
        igs_array = np.array(all_igs)
        np.save(os.path.join(batch_dir, 'all_integrated_gradients.npy'), igs_array)
        
        summary_path = os.path.join(batch_dir, 'batch_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Batch Integrated Gradients Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Samples processed: {start_idx} to {end_idx-1}\n")
            f.write(f"Total samples: {len(batch_results)}\n")
            f.write(f"Integration steps: {steps}\n\n")
            
            f.write("Sample Details:\n")
            f.write("-" * 30 + "\n")
            for i, info in enumerate(all_info):
                f.write(f"Sample {start_idx + i}: Patient {info['patient_id']} - {info['class_name']}\n")
        
        print(f"Batch results saved to {batch_dir}")
        
    except Exception as e:
        print(f"Failed to save batch results: {e}")


def analyze_full_test_set(unet_model, steps=20, batch_size=50):
    """
    Analyze the entire test set in manageable batches
    
    Args:
        unet_model: Trained U-Net model
        steps: Integration steps (reduced for efficiency)
        batch_size: Number of samples per batch
    
    Returns:
        Tuple of (all_results, all_failed_samples)
    """
    print("="*80)
    print("FULL TEST SET INTEGRATED GRADIENTS ANALYSIS")
    print("="*80)
    
    comprehensive_data, _ = load_saved_data()
    total_samples = len(comprehensive_data['X_test'])
    
    print(f"Total test samples: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Integration steps: {steps}")
    
    all_results = []
    all_failed = []
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING BATCH: {start_idx} to {end_idx-1}")
        print(f"{'='*60}")
        
        batch_results, failed_samples = run_batch_integrated_gradients(
            unet_model=unet_model,
            start_idx=start_idx,
            end_idx=end_idx,
            steps=steps,
            target_mode='mean_confidence',
            save_results=True
        )
        
        all_results.extend(batch_results)
        all_failed.extend(failed_samples)
        
        import gc
        gc.collect()
    
    print(f"\n{'='*80}")
    print("FULL DATASET ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(all_results)}")
    print(f"Total failed samples: {len(all_failed)}")
    print(f"Success rate: {len(all_results)/total_samples*100:.1f}%")
    
    return all_results, all_failed


def single_sample_analysis(unet_model):
    """Analyze a single sample with full visualization"""
    try:
        print(f"\n{'='*60}")
        print(f"SINGLE SAMPLE ANALYSIS")
        print(f"{'='*60}")
        
        results = run_integrated_gradients_analysis(
            unet_model=unet_model,
            image_index=0,
            steps=50,
            target_mode='mean_confidence'
        )
        
        print(f"Single sample analysis completed!")
        print(f"Patient ID: {results['sample_info']['patient_id']}")
        print(f"Classification: {results['sample_info']['class_name']}")
        
    except Exception as e:
        print(f"Error during single sample analysis: {e}")


def batch_analysis_example(unet_model):
    """Analyze first 10 samples as demonstration"""
    try:
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS EXAMPLE")
        print(f"{'='*60}")
        
        batch_results, failed = run_batch_integrated_gradients(
            unet_model=unet_model,
            start_idx=0,
            end_idx=10,
            steps=30,
            target_mode='mean_confidence',
            save_results=True
        )
        print(f"Batch analysis completed! Processed {len(batch_results)} samples")
        
    except Exception as e:
        print(f"Error during batch analysis: {e}")


def full_test_set_analysis(unet_model):
    """Analyze the complete test set"""
    try:
        print(f"\n{'='*60}")
        print(f"FULL TEST SET ANALYSIS")
        print(f"{'='*60}")
        
        all_results, all_failed = analyze_full_test_set(
            unet_model=unet_model,
            steps=20,
            batch_size=25
        )
        print(f"Full test set analysis completed!")
        
    except Exception as e:
        print(f"Error during full test set analysis: {e}")


def main():
    """Main execution function with multiple analysis options"""
    
    Config.create_directories()
    
    print("Loading U-Net model...")
    unet_model = load_unet_model()
    
    if unet_model is None:
        print("Cannot proceed without a trained model")
        print("Please ensure your model is saved at: workspace/output/models/best_unet_model.h5")
        return
    
    print("\n" + "="*60)
    print("INTEGRATED GRADIENTS ANALYSIS OPTIONS")
    print("="*60)
    print("Choose analysis mode by uncommenting the desired option:")
    print("1. Single sample analysis (with visualization)")
    print("2. Batch analysis (custom range)")
    print("3. Full test set analysis")
    print("="*60)
    
    # Uncomment one of the following options to run:
    
    single_sample_analysis(unet_model)
    
    # batch_analysis_example(unet_model)
    
    # full_test_set_analysis(unet_model)


if __name__ == "__main__":
    main()