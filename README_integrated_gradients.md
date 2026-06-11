# Integrated Gradients for U-Net XAI Analysis

This is a cleaned-up implementation focused specifically on **Integrated Gradients** for explaining U-Net segmentation models on medical imaging data (LIDC-IDRI dataset).

## Files Overview

- `integrated_gradients_unet.py` - Main implementation of integrated gradients for U-Net
- `example_ig_usage.py` - Example usage and batch processing scripts
- `README_integrated_gradients.md` - This documentation

## Key Features

### 🎯 Focused Implementation
- **Only Integrated Gradients** - Removed other XAI methods for clarity
- **Streamlined code** - Removed unnecessary complexity
- **Memory efficient** - Processes samples individually to avoid memory issues

### 🔬 Core Components

1. **IntegratedGradientsExplainer Class**
   ```python
   explainer = IntegratedGradientsExplainer(model, steps=50)
   integrated_grads, prediction = explainer.explain(input_image)
   ```

2. **Target Modes**
   - `mean_confidence` - Average prediction confidence
   - `max_confidence` - Maximum prediction confidence  
   - `total_area` - Total predicted area

3. **Visualization**
   - Original image and ground truth
   - U-Net predictions
   - Integrated gradients heatmaps
   - Enhanced visualizations with overlays

## Quick Start

### 1. Load Your Model
```python
import tensorflow as tf
from integrated_gradients_unet import run_integrated_gradients_analysis

# Load your trained U-Net model
model = tf.keras.models.load_model('path/to/your/unet_model.h5')
```

### 2. Run Analysis
```python
# Analyze a single sample
results = run_integrated_gradients_analysis(
    unet_model=model,
    image_index=0,  # Which test sample to analyze
    steps=50,       # Number of integration steps
    target_mode='mean_confidence'
)
```

### 3. Batch Analysis
```python
# Analyze multiple samples
for idx in [0, 5, 10, 15, 20]:
    results = run_integrated_gradients_analysis(model, idx)
```

## Configuration

Edit the `Config` class in `integrated_gradients_unet.py`:

```python
class Config:
    DATA_DIR = "/workspace/storage/lidc_dataset"  # Your LIDC dataset path
    OUTPUT_DIR = "workspace/output_latest"        # Output directory
    IMAGE_SIZE = (512, 512)                      # Image dimensions
    IG_STEPS = 50                                # Integration steps
    SEGMENTATION_THRESHOLD = 0.4                 # Prediction threshold
```

## Integration Steps Parameter

- **Fewer steps (20-30)**: Faster computation, less precise
- **More steps (50-100)**: Slower computation, more precise
- **Recommended**: 50 steps for good balance

## Output

The analysis generates:

1. **Visualization plots** showing:
   - Original CT images
   - Ground truth masks
   - U-Net predictions
   - Integrated gradients heatmaps
   - Enhanced overlays

2. **Saved images** in the output directory

3. **Analysis results** dictionary containing:
   - Sample information (patient ID, classification, etc.)
   - Integrated gradients arrays
   - Predictions
   - Original images and ground truth

## Example Usage

```python
# Basic usage
from integrated_gradients_unet import run_integrated_gradients_analysis
from example_ig_usage import load_unet_model

# Load model
model = load_unet_model()

# Analyze sample
results = run_integrated_gradients_analysis(
    unet_model=model,
    image_index=0,
    steps=50,
    target_mode='mean_confidence'
)

print(f"Patient: {results['sample_info']['patient_id']}")
print(f"Classification: {results['sample_info']['class_name']}")
```

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- pydicom
- scikit-image
- tqdm

## Dataset Structure

Expects LIDC-IDRI dataset with:
- DICOM files (.dcm)
- XML annotation files (.xml)
- Saved model outputs (comprehensive_results.npz, detailed_metadata.pkl)

## Memory Considerations

- Processes one sample at a time to avoid memory issues
- Adjustable integration steps for memory/accuracy trade-off
- Automatic garbage collection between samples

## Troubleshooting

1. **Model loading errors**: Check custom_objects in model loading
2. **Memory issues**: Reduce integration steps or image size
3. **Dataset path errors**: Verify DATA_DIR in Config class
4. **Missing files**: Ensure comprehensive_results.npz and detailed_metadata.pkl exist

## Customization

You can easily customize:
- Number of integration steps
- Target computation modes
- Visualization parameters
- Baseline images (default: zeros)
- Output formats and paths