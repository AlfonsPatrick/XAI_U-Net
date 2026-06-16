"""
Export Demo Data for XAI U-Net Web Demo
=======================================

Run this script on the machine where your trained model and test data reside
(e.g., Vast.ai or any machine with TensorFlow + GPU).

It will:
  1. Load the trained model + comprehensive_results.npz + detailed_metadata.pkl
  2. Select a diverse set of test samples (balanced across classes)
  3. Pre-compute Integrated Gradients attributions for each sample
  4. Render all visualization images as PNGs
  5. Save a samples.json manifest with metadata + metrics

Usage:
  python export_demo_data.py --data-dir /workspace/output --output-dir ./data --num-samples 30

Prerequisites:
  pip install tensorflow numpy matplotlib Pillow tqdm
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import argparse
from pathlib import Path
import pickle
from PIL import Image
from tqdm import tqdm

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARNING] TensorFlow not available. IG computation will be skipped.")


# ============================================================================
# Integrated Gradients Explainer (same as in standalone_ig_example.py)
# ============================================================================

class IntegratedGradientsExplainer:
    """Integrated Gradients explainer for U-Net segmentation models"""

    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps

    def explain(self, input_image, baseline=None, target_mode='mean_confidence'):
        """Compute integrated gradients attribution"""
        if not tf.is_tensor(input_image):
            input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

        if baseline is None:
            baseline = tf.zeros_like(input_image)
        elif not tf.is_tensor(baseline):
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        alphas = tf.linspace(0.0, 1.0, self.steps + 1)
        path_gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (input_image - baseline)

            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                prediction = self.model(interpolated)

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

        integrated_grads = tf.reduce_mean(tf.stack(path_gradients), axis=0)
        integrated_grads = integrated_grads * (input_image - baseline)
        final_prediction = self.model(input_image)

        return integrated_grads, final_prediction


# ============================================================================
# Image Rendering Utilities
# ============================================================================

def save_grayscale(array_2d, save_path):
    """Save a normalized [0,1] 2D array as a grayscale PNG"""
    img = (np.clip(array_2d, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(save_path, optimize=True)


def save_colormapped(array_2d, cmap_name, save_path, vmin=None, vmax=None):
    """Save a 2D array as a colormapped RGB PNG"""
    cmap = plt.cm.get_cmap(cmap_name)
    if vmin is None:
        vmin = float(array_2d.min())
    if vmax is None:
        vmax = float(array_2d.max())
    if abs(vmax - vmin) < 1e-10:
        vmax = vmin + 1.0
    normalized = (array_2d - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(save_path, optimize=True)


def save_overlay(base_2d, mask_2d, color_rgb, save_path, alpha=0.45):
    """Save a grayscale image with a colored mask overlay using alpha blending"""
    # Convert grayscale base to RGB float
    base_rgb = np.stack([base_2d] * 3, axis=-1)
    base_rgb = np.clip(base_rgb, 0, 1)

    # Create color layer
    color_layer = np.zeros((*base_2d.shape, 3))
    for c in range(3):
        color_layer[:, :, c] = color_rgb[c]

    # Alpha blend: result = base * (1 - mask_alpha) + color * mask_alpha
    mask_f = mask_2d.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / mask_f.max()
    mask_alpha = mask_f[:, :, np.newaxis] * alpha

    blended = base_rgb * (1 - mask_alpha) + color_layer * mask_alpha
    result = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(result).save(save_path, optimize=True)


def save_ig_heatmap(ig_2d, save_path):
    """Save integrated gradients as a diverging heatmap (red=positive, blue=negative)"""
    vmax = max(abs(float(ig_2d.min())), abs(float(ig_2d.max())))
    if vmax < 1e-10:
        vmax = 1.0
    save_colormapped(ig_2d, 'RdBu_r', save_path, vmin=-vmax, vmax=vmax)


def save_ig_overlay(base_2d, ig_2d, save_path, enhancement=500):
    """Save CT image with IG attribution overlay"""
    ig_magnitude = np.abs(ig_2d) * enhancement
    ig_magnitude = np.clip(ig_magnitude, 0, 1)
    save_overlay(base_2d, ig_magnitude, (0.0, 0.95, 0.85), save_path, alpha=0.55)


# ============================================================================
# Sample Selection
# ============================================================================

def select_diverse_samples(y_test, test_metadata, num_samples=30):
    """
    Select a diverse set of test samples with balanced class representation.
    Prioritizes samples with nodules (non-zero ground truth area) for more
    interesting visualizations.
    """
    class_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(y_test):
        class_indices[int(label)].append(i)

    per_class = num_samples // 3
    remainder = num_samples % 3

    selected = []
    for class_label in sorted(class_indices.keys()):
        indices = class_indices[class_label]
        n = per_class + (1 if class_label < remainder else 0)
        n = min(n, len(indices))

        if n > 0:
            np.random.seed(42 + class_label)
            chosen = np.random.choice(indices, n, replace=False)
            selected.extend(chosen.tolist())

    return sorted(selected)


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(pred_2d, gt_2d, threshold=0.4):
    """Compute segmentation metrics between prediction and ground truth"""
    pred_binary = (pred_2d > threshold).astype(np.uint8)
    gt_binary = (gt_2d > 0.5).astype(np.uint8)

    intersection = int(np.sum(pred_binary * gt_binary))
    pred_area = int(np.sum(pred_binary))
    gt_area = int(np.sum(gt_binary))
    union = pred_area + gt_area - intersection

    dice = (2 * intersection) / (pred_area + gt_area + 1e-6)
    iou_val = intersection / (union + 1e-6)
    precision = intersection / (pred_area + 1e-6)
    recall = intersection / (gt_area + 1e-6)

    return {
        'dice': round(float(dice), 4),
        'iou': round(float(iou_val), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'gt_area': gt_area,
        'pred_area': pred_area,
        'intersection': intersection,
        'max_confidence': round(float(np.max(pred_2d)), 4),
        'mean_confidence': round(float(np.mean(pred_2d)), 6),
    }


# ============================================================================
# Main Export Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export demo data for the XAI U-Net web demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full export with IG computation (requires GPU + TensorFlow):
  python export_demo_data.py --data-dir /workspace/output --output-dir ./data

  # Export without IG (faster, no model needed):
  python export_demo_data.py --data-dir /workspace/output --output-dir ./data --skip-ig

  # Export fewer samples:
  python export_demo_data.py --data-dir /workspace/output --output-dir ./data --num-samples 15
        """
    )
    parser.add_argument('--data-dir', type=str, default=r'd:\docs\UGM\KULIYEAH\skripsi fix\output',
                        help='Directory containing comprehensive_results.npz, '
                             'detailed_metadata.pkl, and models/ (default: d:\\docs\\UGM\\KULIYEAH\\skripsi fix\\output)')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for demo data (default: ./data)')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='Number of test samples to export (default: 30)')
    parser.add_argument('--ig-steps', type=int, default=50,
                        help='Number of integration steps for IG (default: 50)')
    parser.add_argument('--skip-ig', action='store_true',
                        help='Skip IG computation (useful if no GPU/model available)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    comprehensive_path = os.path.join(args.data_dir, 'comprehensive_results.npz')
    if not os.path.exists(comprehensive_path):
        print(f"[ERROR] File not found: {comprehensive_path}")
        sys.exit(1)
    data = np.load(comprehensive_path)

    metadata_path = os.path.join(args.data_dir, 'detailed_metadata.pkl')
    if not os.path.exists(metadata_path):
        print(f"[ERROR] File not found: {metadata_path}")
        sys.exit(1)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    X_test = data['X_test']
    y_test = data['y_test']
    m_test = data['m_test']
    pred_masks = data['pred_masks']
    test_metadata = metadata['test_metadata']

    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    print(f"Test set: {len(X_test)} samples")
    for label, name in class_names.items():
        count = int(np.sum(y_test == label))
        print(f"  {name}: {count}")

    # ── Select Samples ─────────────────────────────────────────────────────
    selected_indices = select_diverse_samples(y_test, test_metadata, args.num_samples)
    print(f"\nSelected {len(selected_indices)} diverse samples for export")

    # ── Load Model (if computing IG) ───────────────────────────────────────
    model = None
    if not args.skip_ig and HAS_TF:
        model_path = os.path.join(args.data_dir, 'models', 'best_unet_model.h5')
        if os.path.exists(model_path):
            print(f"\nLoading model from {model_path}...")
            custom_objects = {
                'dice_coefficient': lambda y_true, y_pred: tf.constant(0.0),
                'dice_loss': lambda y_true, y_pred: tf.constant(0.0),
                'iou': lambda y_true, y_pred: tf.constant(0.0),
                'focal_tversky_loss': lambda y_true, y_pred: tf.constant(0.0)
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully")
        else:
            print(f"[WARNING] Model not found at {model_path}, skipping IG computation")
    elif args.skip_ig:
        print("\nSkipping IG computation (--skip-ig flag)")
    else:
        print("\nSkipping IG computation (TensorFlow not available)")

    # ── Process Samples ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROCESSING SAMPLES")
    print("=" * 60)

    samples_manifest = []

    for i, idx in enumerate(selected_indices):
        idx = int(idx)
        print(f"\n[{i + 1}/{len(selected_indices)}] Sample index {idx}")

        sample_dir = images_dir / f'sample_{i:02d}'
        sample_dir.mkdir(exist_ok=True)

        # Extract data
        image = np.squeeze(X_test[idx])
        pred = np.squeeze(pred_masks[idx])
        gt = np.squeeze(m_test[idx])
        class_label = int(y_test[idx])
        meta = test_metadata[idx]

        # 1. CT Image (grayscale)
        save_grayscale(image, sample_dir / 'ct.png')

        # 2. Prediction heatmap (inferno colormap)
        save_colormapped(pred, 'inferno', sample_dir / 'prediction.png', vmin=0, vmax=1)

        # 3. Prediction overlay (green on CT)
        pred_binary = (pred > 0.4).astype(np.float32)
        save_overlay(image, pred_binary, (0.18, 0.95, 0.32), sample_dir / 'pred_overlay.png', alpha=0.45)

        # 4. Ground truth overlay (blue on CT)
        gt_float = gt.astype(np.float32)
        save_overlay(image, gt_float, (0.22, 0.55, 1.0), sample_dir / 'gt_overlay.png', alpha=0.45)

        # 5. Integrated Gradients (if model available)
        has_ig = False
        ig_stats = {}
        if model is not None:
            try:
                explainer = IntegratedGradientsExplainer(model, steps=args.ig_steps)
                input_tensor = X_test[idx:idx + 1]
                ig, _ = explainer.explain(input_tensor, target_mode='mean_confidence')
                ig_2d = np.squeeze(ig.numpy())

                # IG heatmap (diverging colormap)
                save_ig_heatmap(ig_2d, sample_dir / 'ig_heatmap.png')

                # IG overlay on CT
                save_ig_overlay(image, ig_2d, sample_dir / 'ig_overlay.png', enhancement=500)

                has_ig = True
                ig_stats = {
                    'ig_min': round(float(np.min(ig_2d)), 6),
                    'ig_max': round(float(np.max(ig_2d)), 6),
                    'ig_mean': round(float(np.mean(ig_2d)), 6),
                    'ig_std': round(float(np.std(ig_2d)), 6),
                }
                print(f"  ✓ IG computed | range=[{ig_stats['ig_min']:.2e}, {ig_stats['ig_max']:.2e}]")
            except Exception as e:
                print(f"  ✗ IG failed: {e}")

        # Compute metrics
        metrics = compute_metrics(pred, gt)

        # Build manifest entry
        sample_entry = {
            'id': i,
            'original_index': idx,
            'patient_id': int(meta.get('patient_id', 0)),
            'slice_idx': int(meta.get('slice_idx', 0)),
            'class_label': class_label,
            'class_name': class_names.get(class_label, 'Unknown'),
            'malignancy_score': int(meta.get('malignancy', 0)),
            'has_nodule': bool(meta.get('has_nodule', False)),
            'metrics': metrics,
            'has_ig': has_ig,
            'ig_stats': ig_stats,
            'images': {
                'ct': f'images/sample_{i:02d}/ct.png',
                'prediction': f'images/sample_{i:02d}/prediction.png',
                'pred_overlay': f'images/sample_{i:02d}/pred_overlay.png',
                'gt_overlay': f'images/sample_{i:02d}/gt_overlay.png',
            }
        }

        if has_ig:
            sample_entry['images']['ig_heatmap'] = f'images/sample_{i:02d}/ig_heatmap.png'
            sample_entry['images']['ig_overlay'] = f'images/sample_{i:02d}/ig_overlay.png'

        samples_manifest.append(sample_entry)
        print(f"  Patient {meta.get('patient_id', '?')}, {class_names.get(class_label, '?')}, "
              f"Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}")

    # ── Save Manifest ──────────────────────────────────────────────────────
    # Compute aggregate statistics
    all_dice = [s['metrics']['dice'] for s in samples_manifest]
    all_iou = [s['metrics']['iou'] for s in samples_manifest]

    manifest = {
        'project': 'XAI U-Net: Explainable Lung Nodule Segmentation',
        'description': 'Interactive demo of Attention U-Net with Integrated Gradients '
                       'for lung nodule segmentation and classification on LIDC-IDRI CT scans.',
        'dataset': 'LIDC-IDRI',
        'model': 'Attention U-Net',
        'xai_method': 'Integrated Gradients',
        'image_size': [512, 512],
        'num_patients_trained': 100,
        'ig_steps': args.ig_steps,
        'segmentation_threshold': 0.4,
        'total_samples': len(samples_manifest),
        'aggregate_metrics': {
            'mean_dice': round(float(np.mean(all_dice)), 4),
            'mean_iou': round(float(np.mean(all_iou)), 4),
            'std_dice': round(float(np.std(all_dice)), 4),
            'std_iou': round(float(np.std(all_iou)), 4),
        },
        'class_distribution': {
            'Normal': sum(1 for s in samples_manifest if s['class_label'] == 0),
            'Benign': sum(1 for s in samples_manifest if s['class_label'] == 1),
            'Malignant': sum(1 for s in samples_manifest if s['class_label'] == 2),
        },
        'samples': samples_manifest,
    }

    manifest_path = output_dir / 'samples.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output directory:  {output_dir.resolve()}")
    print(f"Manifest file:     {manifest_path}")
    print(f"Total samples:     {len(samples_manifest)}")
    print(f"Samples with IG:   {sum(1 for s in samples_manifest if s['has_ig'])}")
    print(f"Mean Dice:         {manifest['aggregate_metrics']['mean_dice']:.4f}")
    print(f"Mean IoU:          {manifest['aggregate_metrics']['mean_iou']:.4f}")
    print(f"\nClass distribution:")
    for cls, count in manifest['class_distribution'].items():
        print(f"  {cls}: {count}")
    print(f"\nNext steps:")
    print(f"  1. Copy the '{output_dir}' folder into your demo/ directory")
    print(f"  2. Open demo/index.html in a browser (or serve with a local server)")
    print(f"  3. Deploy to GitHub Pages for public access")


if __name__ == '__main__':
    main()
