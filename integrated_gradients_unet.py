import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import warnings
warnings.filterwarnings('ignore')

# Medical imaging imports
import pydicom
import xml.etree.ElementTree as ET
from skimage.draw import polygon
from tqdm import tqdm

# TensorFlow imports
import tensorflow as tf

print("Integrated Gradients U-Net XAI - Imports completed successfully")


class Config:
    """Configuration parameters for integrated gradients analysis"""
    # Dataset and output paths
    DATA_DIR = "/workspace/storage/lidc_dataset"
    OUTPUT_DIR = "workspace/output_latest"
    
    # Image processing parameters
    IMAGE_SIZE = (512, 512)
    HU_MIN = -1000  # Minimum Hounsfield Unit value
    HU_MAX = 400    # Maximum Hounsfield Unit value
    
    # Analysis parameters
    SEGMENTATION_THRESHOLD = 0.4  # Threshold for binary segmentation
    IG_STEPS = 50                 # Default number of integration steps
    
    @classmethod
    def create_directories(cls):
        """Create necessary output directories if they don't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


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


def load_dicom_series(dicom_files):
    """
    Load DICOM series with error handling and proper sorting
    
    Args:
        dicom_files: List of DICOM file paths
        
    Returns:
        List of dictionaries containing image data and metadata
    """
    if not dicom_files:
        return []
    
    loaded_data = []
    for filepath in tqdm(dicom_files, desc="Loading DICOM files"):
        try:
            dcm = pydicom.dcmread(filepath)
            img = dcm.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept if available for proper HU values
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                img = img * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Extract metadata for proper slice ordering
            slice_location = getattr(dcm, 'SliceLocation', 0)
            instance_number = getattr(dcm, 'InstanceNumber', 0)
            
            loaded_data.append({
                'image': img,
                'uid': dcm.SOPInstanceUID,
                'filepath': filepath,
                'slice_location': slice_location,
                'instance_number': instance_number
            })
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    # Sort by slice location and instance number for proper ordering
    loaded_data.sort(key=lambda x: (x['slice_location'], x['instance_number']))
    return loaded_data


def parse_lidc_xml(xml_path):
    """
    Parse LIDC XML annotations to extract nodule information
    
    Args:
        xml_path: Path to LIDC XML annotation file
        
    Returns:
        Dictionary mapping SOP Instance UIDs to nodule data
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Extract namespace for proper XML parsing
        ns = {'idri': root.tag.split('}')[0].strip('{')}
        
        nodule_data = {}
        
        # Parse each reading session
        for session in root.findall('.//idri:readingSession', ns):
            for nodule in session.findall('.//idri:unblindedReadNodule', ns):
                
                # Extract nodule characteristics (malignancy, subtlety, etc.)
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
                
                # Extract ROI polygon coordinates for each slice
                for roi in nodule.findall('.//idri:roi', ns):
                    sop_elem = roi.find('idri:imageSOP_UID', ns)
                    if sop_elem is None or sop_elem.text is None:
                        continue
                    
                    uid = sop_elem.text.strip()
                    
                    # Extract polygon coordinates from edge map
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
                    
                    # Store nodule data if valid polygon found
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


def create_mask_from_nodules(nodule_data, image_shape):
    """Create binary mask from nodule annotations"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    malignancy_scores = []
    
    for nodule_info in nodule_data:
        polygon_coords = nodule_info['polygon']
        characteristics = nodule_info['characteristics']
        
        if len(polygon_coords) >= 3:
            rr, cc = polygon(
                [p[0] for p in polygon_coords], 
                [p[1] for p in polygon_coords], 
                image_shape
            )
            mask[rr, cc] = 1
            
            malignancy = characteristics.get('malignancy', 0)
            malignancy_scores.append(malignancy)
    
    final_malignancy = max(malignancy_scores) if malignancy_scores else 0
    return mask, final_malignancy


def load_saved_data():
    """Load the saved training data and metadata"""
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
    
    Implements the integrated gradients method using Riemann sum approximation
    to explain U-Net predictions for medical image segmentation.
    """
    
    def __init__(self, model, steps=50):
        """
        Initialize the integrated gradients explainer
        
        Args:
            model: Trained U-Net model for segmentation
            steps: Number of integration steps for Riemann sum approximation
        """
        self.model = model
        self.steps = steps
    
    def explain(self, input_image, baseline=None, target_mode='mean_confidence'):
        """
        Generate integrated gradients explanation for input image
        
        Implements the formula:
        IntegratedGrads_i(x) = (x_i - x'_i) × ∫[α=0 to 1] (∂F(x' + α×(x - x'))/∂x_i) dα
        
        Args:
            input_image: Input image tensor with shape (batch_size=1, height, width, channels)
            baseline: Baseline image (default: zeros). Should have same shape as input_image
            target_mode: Method to compute scalar target from segmentation output:
                        'mean_confidence' - average prediction confidence
                        'max_confidence' - maximum prediction confidence  
                        'total_area' - sum of all predictions (total predicted area)
        
        Returns:
            integrated_grads: Integrated gradients with same shape as input_image
            prediction: Final model prediction for the input image
        """
        # Step 1: Ensure input is proper tensor format
        if not tf.is_tensor(input_image):
            input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        
        # Step 2: Create baseline (x' in the formula)
        if baseline is None:
            baseline = tf.zeros_like(input_image)  # Default: zero baseline
        elif not tf.is_tensor(baseline):
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
        
        # Step 3: Generate interpolation path for Riemann sum
        # Creates points α = [0, 1/m, 2/m, ..., 1] for m steps
        alphas = tf.linspace(0.0, 1.0, self.steps + 1)
        
        # Step 4: Compute gradients along the interpolation path
        path_gradients = []
        
        for alpha in tqdm(alphas, desc="Computing integrated gradients"):
            # Step 4a: Create interpolated point x' + α×(x - x')
            interpolated = baseline + alpha * (input_image - baseline)
            
            # Step 4b: Compute gradient at this interpolated point
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                prediction = self.model(interpolated)  # F(x' + α×(x - x'))
                
                # Step 4c: Convert segmentation output to scalar target
                if target_mode == 'mean_confidence':
                    target = tf.reduce_mean(prediction)
                elif target_mode == 'max_confidence':
                    target = tf.reduce_max(prediction)
                elif target_mode == 'total_area':
                    target = tf.reduce_sum(prediction)
                else:
                    target = tf.reduce_mean(prediction)
            
            # Compute ∂F(...)/∂x_i at this interpolation point
            gradient = tape.gradient(target, interpolated)
            path_gradients.append(gradient)
        
        # Step 5: Approximate integral using Riemann sum (average of gradients)
        # This approximates: ∫[α=0 to 1] (∂F(...)/∂x_i) dα ≈ (1/m) × Σ gradients
        integrated_grads = tf.reduce_mean(tf.stack(path_gradients), axis=0)
        
        # Step 6: Apply final scaling factor (x_i - x'_i)
        integrated_grads = integrated_grads * (input_image - baseline)
        
        # Get final prediction for return
        final_prediction = self.model(input_image)
        
        return integrated_grads, final_prediction
    
    def explain_batch(self, input_batch, baseline=None, target_mode='mean_confidence'):
        """
        Explain a batch of images with memory-efficient processing
        
        Processes images one by one to avoid memory issues with large batches
        
        Args:
            input_batch: Batch of input images with shape (batch_size, height, width, channels)
            baseline: Baseline image or batch of baselines (default: zeros)
            target_mode: Target computation mode (same options as explain method)
        
        Returns:
            List of (integrated_grads, prediction) tuples for each image in batch
        """
        results = []
        
        for i in range(input_batch.shape[0]):
            # Process each image individually to manage memory usage
            single_input = input_batch[i:i+1]
            single_baseline = baseline[i:i+1] if baseline is not None else None
            
            ig, pred = self.explain(single_input, single_baseline, target_mode)
            results.append((ig, pred))
        
        return results


def get_ground_truth_mask(sample_metadata):
    """Retrieve actual ground truth mask for a sample"""
    try:
        patient_id = sample_metadata['patient_id']
        patient_dir = patient_root(patient_id)
        
        # Load DICOM files
        dicom_files = [os.path.join(r, f) for r, _, files in os.walk(patient_dir) 
                      for f in files if f.endswith('.dcm')]
        
        if not dicom_files:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
        
        dicom_data = load_dicom_series(dicom_files)
        if not dicom_data:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
        
        # Find the specific slice
        uid_to_idx = {data['uid']: idx for idx, data in enumerate(dicom_data)}
        target_uid = sample_metadata['uid']
        
        if target_uid not in uid_to_idx:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
        
        slice_idx = uid_to_idx[target_uid]
        original_image = dicom_data[slice_idx]['image']
        
        # Load XML annotations
        xml_files = [os.path.join(r, f) for r, _, files in os.walk(patient_dir) 
                    for f in files if f.endswith('.xml')]
        
        nodule_data = {}
        for xml_file in xml_files:
            nodule_data.update(parse_lidc_xml(xml_file))
        
        # Create mask
        if target_uid in nodule_data:
            mask, _ = create_mask_from_nodules(nodule_data[target_uid], original_image.shape)
            mask_resized = cv2.resize(mask, Config.IMAGE_SIZE)
            return mask_resized.astype(np.uint8)
        else:
            return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
            
    except Exception as e:
        print(f"Error retrieving ground truth: {e}")
        return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)


def visualize_integrated_gradients(image, ground_truth, prediction, integrated_grads, 
                                 sample_info, save_path=None):
    """
    Visualize integrated gradients results
    
    Args:
        image: Original input image
        ground_truth: Ground truth mask
        prediction: Model prediction
        integrated_grads: Integrated gradients
        sample_info: Sample metadata
        save_path: Optional path to save the figure
    """
    # Prepare data
    image_2d = np.squeeze(image)
    gt_2d = np.squeeze(ground_truth)
    pred_2d = np.squeeze(prediction)
    ig_2d = np.squeeze(integrated_grads)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original data
    axes[0, 0].imshow(image_2d, cmap='gray')
    axes[0, 0].set_title('Original CT Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_2d, cmap='hot')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_2d, cmap='hot')
    axes[0, 2].set_title('U-Net Prediction')
    axes[0, 2].axis('off')
    
    # Prediction overlay
    pred_binary = (pred_2d > Config.SEGMENTATION_THRESHOLD).astype(np.uint8)
    overlay = np.zeros((*image_2d.shape, 3))
    overlay[:, :, 0] = image_2d
    overlay[:, :, 1] = pred_binary
    axes[0, 3].imshow(overlay)
    axes[0, 3].set_title('Image + Prediction')
    axes[0, 3].axis('off')
    
    # Row 2: Integrated gradients analysis
    axes[1, 0].imshow(ig_2d, cmap='RdBu_r')
    axes[1, 0].set_title('Integrated Gradients')
    axes[1, 0].axis('off')
    
    # Enhanced IG for better visibility
    ig_enhanced = ig_2d * 1000
    axes[1, 1].imshow(ig_enhanced, cmap='RdBu_r')
    axes[1, 1].set_title('Enhanced IG (x1000)')
    axes[1, 1].axis('off')
    
    # IG overlay on original image
    ig_overlay = np.zeros((*image_2d.shape, 3))
    ig_overlay[:, :, 0] = image_2d
    ig_overlay[:, :, 1] = np.abs(ig_enhanced)
    ig_overlay[:, :, 2] = np.abs(ig_enhanced)
    axes[1, 2].imshow(ig_overlay)
    axes[1, 2].set_title('Image + IG Overlay')
    axes[1, 2].axis('off')
    
    # Statistics
    stats_text = f"""
SAMPLE INFO:
Patient ID: {sample_info.get('patient_id', 'N/A')}
Class: {sample_info.get('class_name', 'N/A')}
Malignancy: {sample_info.get('malignancy_score', 'N/A')}

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
    
    axes[1, 3].text(0.05, 0.95, stats_text, transform=axes[1, 3].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[1, 3].axis('off')
    
    # Set title
    plt.suptitle(f'Integrated Gradients Analysis - Patient {sample_info.get("patient_id", "N/A")} '
                f'({sample_info.get("class_name", "Unknown")})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def run_integrated_gradients_analysis(unet_model, image_index, steps=50, target_mode='mean_confidence'):
    """
    Run complete integrated gradients analysis for a specific test image
    
    This function loads the test data, runs integrated gradients analysis,
    and creates visualizations with comprehensive results.
    
    Args:
        unet_model: Trained U-Net model for segmentation
        image_index: Index of the test image to analyze
        steps: Number of integration steps for accuracy vs speed tradeoff
        target_mode: Method to compute target from segmentation output
    
    Returns:
        dict: Complete analysis results containing:
            - sample_info: Patient metadata and classification
            - integrated_grads: Computed integrated gradients array
            - prediction: Model prediction
            - image: Original input image
            - ground_truth: Ground truth segmentation mask
    """
    print("="*60)
    print("INTEGRATED GRADIENTS U-NET ANALYSIS")
    print("="*60)
    
    # Load preprocessed test data and metadata
    comprehensive_data, detailed_metadata = load_saved_data()
    
    # Extract sample information and metadata
    test_metadata = detailed_metadata['test_metadata']
    sample_metadata = test_metadata[image_index]
    
    # Map classification labels to readable names
    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    class_label = comprehensive_data['y_test'][image_index]
    
    # Compile sample information for analysis
    sample_info = {
        'patient_id': sample_metadata['patient_id'],
        'uid': sample_metadata['uid'],
        'slice_idx': sample_metadata['slice_idx'],
        'class_label': class_label,
        'class_name': class_names.get(class_label, 'Unknown'),
        'malignancy_score': sample_metadata.get('malignancy', 0)
    }
    
    print(f"Analyzing Patient {sample_info['patient_id']} - {sample_info['class_name']}")
    
    # Extract test data for the specified image
    image_to_explain = comprehensive_data['X_test'][image_index:image_index+1]
    unet_prediction = comprehensive_data['pred_masks'][image_index:image_index+1]
    
    # Retrieve actual ground truth mask from DICOM/XML data
    ground_truth = get_ground_truth_mask(sample_metadata)
    ground_truth = ground_truth[np.newaxis, :, :, np.newaxis]  # Add batch and channel dimensions
    
    print(f"Data shapes - Image: {image_to_explain.shape}, Ground Truth: {ground_truth.shape}")
    
    # Initialize integrated gradients explainer
    explainer = IntegratedGradientsExplainer(unet_model, steps=steps)
    
    # Compute integrated gradients explanation
    print(f"Computing integrated gradients with {steps} steps...")
    integrated_grads, prediction = explainer.explain(
        image_to_explain, 
        target_mode=target_mode
    )
    
    # Create comprehensive visualization
    print("Creating visualization...")
    save_path = os.path.join(Config.OUTPUT_DIR, f'ig_analysis_patient_{sample_info["patient_id"]}.png')
    
    visualize_integrated_gradients(
        image=image_to_explain,
        ground_truth=ground_truth,
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
        'image': image_to_explain,
        'ground_truth': ground_truth
    }


if __name__ == "__main__":
    # Example usage
    print("Integrated Gradients U-Net XAI Analysis")
    print("Load your trained U-Net model and call:")
    print("run_integrated_gradients_analysis(model, image_index=0)")