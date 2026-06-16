"""
Build Demo Data from Local Output Directory
============================================

Reads the pre-computed results already in your local output/ directory
and generates the demo/data/ folder (images + samples.json manifest).

No TensorFlow, no pydicom, no GPU needed — just numpy + Pillow + matplotlib.

Usage (run from inside the demo/ folder):
    python build_demo_data.py

Or with explicit paths:
    python build_demo_data.py --output-dir "d:\\path\\to\\output" --dest-dir "./data"
"""

import os
import sys
import json
import argparse
import re
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARNING] matplotlib not available. Colormapped images will be skipped.")


# ═══════════════════════════════════════════════════════════════════════════
# Image Rendering Helpers
# ═══════════════════════════════════════════════════════════════════════════

def to_uint8(arr):
    """Normalize a 2D float array to [0, 255] uint8"""
    arr = np.squeeze(arr).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if abs(hi - lo) < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def save_grayscale(arr_2d, path):
    """Save a 2D array as grayscale PNG (normalized)"""
    Image.fromarray(to_uint8(arr_2d), mode='L').save(path, optimize=True)


def save_colormapped(arr_2d, cmap_name, path, vmin=None, vmax=None):
    """Save a 2D array as a colormapped RGB PNG using matplotlib"""
    if not HAS_MPL:
        save_grayscale(arr_2d, path)
        return
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    arr = np.squeeze(arr_2d).astype(np.float32)
    _vmin = float(arr.min()) if vmin is None else vmin
    _vmax = float(arr.max()) if vmax is None else vmax
    if abs(_vmax - _vmin) < 1e-9:
        _vmax = _vmin + 1.0
    norm = np.clip((arr - _vmin) / (_vmax - _vmin), 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path, optimize=True)


def save_overlay(base_gray, mask, color_rgb, path, alpha=0.45):
    """Overlay a coloured mask on a grayscale CT image"""
    base = np.squeeze(base_gray).astype(np.float32)
    base_norm = (base - base.min()) / max(base.max() - base.min(), 1e-9)

    base_rgb = np.stack([base_norm] * 3, axis=-1)

    color_layer = np.zeros((*base_norm.shape, 3), dtype=np.float32)
    for c in range(3):
        color_layer[:, :, c] = color_rgb[c]

    mask_f = np.squeeze(mask).astype(np.float32)
    if mask_f.max() > 0:
        mask_f = mask_f / mask_f.max()
    blend_alpha = mask_f[:, :, np.newaxis] * alpha

    blended = base_rgb * (1.0 - blend_alpha) + color_layer * blend_alpha
    Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8)).save(path, optimize=True)


def save_ig_heatmap(ig_2d, path):
    """Save IG array as a Blues heatmap"""
    arr = np.abs(np.squeeze(ig_2d).astype(np.float32))
    vmax = float(arr.max())
    if vmax < 1e-12:
        vmax = 1.0
    save_colormapped(arr, 'Blues', path, vmin=0, vmax=vmax)


def save_ig_overlay(base_gray, ig_2d, path, threshold=0.15):
    """Save CT + IG magnitude overlay (teal highlight on hotspots)"""
    mag = np.abs(np.squeeze(ig_2d).astype(np.float32))
    mag_max = mag.max()
    if mag_max > 1e-12:
        mag = mag / mag_max  # Normalize so max magnitude is exactly 1.0
        
    mask = np.clip(mag, 0, 1)
    save_overlay(base_gray, (mask > threshold).astype(np.float32) * mask,
                 (0.0, 0.95, 0.85), path, alpha=0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(pred, gt, threshold=0.4):
    pred_b = (np.squeeze(pred) > threshold).astype(np.int32)
    gt_b   = (np.squeeze(gt) > 0.5).astype(np.int32)

    inter = int(np.sum(pred_b * gt_b))
    pred_a = int(np.sum(pred_b))
    gt_a   = int(np.sum(gt_b))
    union  = pred_a + gt_a - inter

    dice      = (2 * inter) / (pred_a + gt_a + 1e-6)
    iou_val   = inter / (union + 1e-6)
    precision = inter / (pred_a + 1e-6)
    recall    = inter / (gt_a + 1e-6)

    return {
        'dice': round(float(dice), 4),
        'iou': round(float(iou_val), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'gt_area': gt_a,
        'pred_area': pred_a,
        'intersection': inter,
        'max_confidence': round(float(np.max(pred)), 4),
        'mean_confidence': round(float(np.mean(pred)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Parse sample_info.txt
# ═══════════════════════════════════════════════════════════════════════════

def parse_sample_info(txt_path):
    """Read a sample_info.txt file into a dict"""
    info = {}
    if not os.path.exists(txt_path):
        return info
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                info[key.strip()] = val.strip()
    return info


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Build demo data from local output directory')
    parser.add_argument('--output-dir', type=str,
                        default=r'd:\docs\UGM\KULIYEAH\skripsi fix\output',
                        help='Path to the output/ directory with your training results')
    parser.add_argument('--dest-dir', type=str,
                        default=r'd:\docs\UGM\KULIYEAH\skripsi fix\XAI_U-Net\demo\data',
                        help='Destination for demo data (default: demo/data)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dest_dir   = Path(args.dest_dir)
    images_dir = dest_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load comprehensive_results.npz ────────────────────────────────
    npz_path = output_dir / 'comprehensive_results.npz'
    print(f"Loading {npz_path} ...")
    if not npz_path.exists():
        print(f"[ERROR] Not found: {npz_path}")
        sys.exit(1)
    data = np.load(str(npz_path))

    # Print available keys so we know what's in the file
    print(f"  Keys: {list(data.keys())}")

    X_test     = data['X_test']      # shape (N, H, W, 1) or (N, H, W)
    y_test     = data['y_test']      # class labels (N,)
    pred_masks = data['pred_masks']  # shape (N, H, W, 1) or (N, H, W)
    # Ground truth masks — try common key names
    gt_masks   = None
    for key in ('m_test', 'masks_test', 'gt_masks', 'y_masks', 'test_masks'):
        if key in data:
            gt_masks = data[key]
            print(f"  Using '{key}' as ground truth masks")
            break
    if gt_masks is None:
        print("[WARNING] No ground truth mask array found in npz. GT overlays will be blank.")
        gt_masks = np.zeros_like(pred_masks)

    n_total = len(X_test)
    print(f"  X_test:     {X_test.shape}")
    print(f"  y_test:     {y_test.shape}  (classes)")
    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  gt_masks:   {gt_masks.shape}")

    # ── 2. Extract metadata directly from the NPZ ─────────────────────────
    # The NPZ already contains all per-sample metadata as arrays —
    # no need to load detailed_metadata.pkl (which has pydicom objects).
    print("\nExtracting metadata from NPZ arrays...")

    def _npz_col(key, default=None):
        """Safely retrieve an array column from the NPZ data dict"""
        if key in data:
            return data[key]
        return default

    npz_patient_ids      = _npz_col('test_patient_ids')
    npz_slice_indices    = _npz_col('test_slice_indices')
    npz_uids             = _npz_col('test_uids')
    npz_class_labels     = _npz_col('test_class_labels')
    npz_malignancy_scores= _npz_col('test_malignancy_scores')

    # Mapping for string class labels to integers
    _str_to_int = {'normal': 0, 'benign': 1, 'malignant': 2}

    def _label_to_int(val):
        """Convert a class label value (int, np.int*, or string) to Python int."""
        s = str(val).strip().lower()
        if s in _str_to_int:
            return _str_to_int[s]
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return 0

    def get_meta(idx):
        """Return a metadata dict for test index idx, built from NPZ arrays."""
        m = {}
        if npz_patient_ids        is not None: m['patient_id']      = npz_patient_ids[idx]
        if npz_slice_indices      is not None: m['slice_idx']        = npz_slice_indices[idx]
        if npz_uids               is not None: m['uid']              = npz_uids[idx]
        if npz_class_labels       is not None: m['class_label']      = _label_to_int(npz_class_labels[idx])
        if npz_malignancy_scores  is not None: m['malignancy_score'] = int(npz_malignancy_scores[idx])
        m['has_nodule'] = m.get('class_label', int(y_test[idx])) > 0
        return m

    print(f"  Patient IDs available:       {npz_patient_ids is not None}")
    print(f"  Slice indices available:     {npz_slice_indices is not None}")
    print(f"  Malignancy scores available: {npz_malignancy_scores is not None}")

    # ── 3. Discover ig_sample_* directories ──────────────────────────────
    ig_dirs = sorted(output_dir.glob('ig_sample_*'),
                     key=lambda p: int(re.search(r'\d+', p.name).group()))
    print(f"\nFound {len(ig_dirs)} ig_sample_* directories")

    # Build map: sample_index → ig_array (or None)
    ig_map = {}   # test_index → np.ndarray (H, W)
    ig_info_map = {}   # test_index → info dict from sample_info.txt

    for ig_dir in ig_dirs:
        info = parse_sample_info(ig_dir / 'sample_info.txt')
        npy  = ig_dir / 'integrated_gradients.npy'

        if not npy.exists():
            continue

        ig_arr = np.load(str(npy))
        ig_arr = np.squeeze(ig_arr)

        # The ig_sample_* directories are numbered 0..N — map them directly
        dir_idx = int(re.search(r'\d+', ig_dir.name).group())
        ig_map[dir_idx]      = ig_arr
        ig_info_map[dir_idx] = info

    # ── 4. Decide which test samples to include ──────────────────────────
    # Use the ig_sample_* indices directly (they tell us which test samples
    # were processed). Fall back to selecting the first min(30, n_total).
    if ig_map:
        sample_indices = sorted(ig_map.keys())[:50]   # cap at 50
        print(f"Using {len(sample_indices)} samples matching ig_sample_* directories")
    else:
        # No IG: pick up to 30 balanced samples
        class_names_map = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
        per_class = {}
        for i in range(n_total):
            lbl = int(y_test[i])
            per_class.setdefault(lbl, []).append(i)
        sample_indices = []
        for lbl in sorted(per_class):
            idxs = per_class[lbl][:10]
            sample_indices.extend(idxs)
        sample_indices = sorted(sample_indices)
        print(f"No IG dirs — using {len(sample_indices)} balanced samples")

    class_names_map = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}

    # ── 5. Process each sample ───────────────────────────────────────────
    samples_manifest = []

    for out_idx, test_idx in enumerate(sample_indices):
        test_idx = int(test_idx)
        print(f"  [{out_idx+1}/{len(sample_indices)}] test_idx={test_idx}", end='  ')

        sample_out_dir = images_dir / f'sample_{out_idx:02d}'
        sample_out_dir.mkdir(exist_ok=True)

        # Extract arrays
        img  = np.squeeze(X_test[test_idx])
        pred = np.squeeze(pred_masks[test_idx])
        gt   = np.squeeze(gt_masks[test_idx])
        lbl  = int(y_test[test_idx])
        meta = get_meta(test_idx)

        # Pull metadata fields
        patient_id       = meta.get('patient_id', 0)
        slice_idx        = meta.get('slice_idx', 0)
        malignancy_score = meta.get('malignancy_score', meta.get('malignancy', 0))
        has_nodule       = bool(meta.get('has_nodule', lbl > 0))

        # IG data
        ig_arr  = ig_map.get(out_idx, None)  # prefer matching by out_idx
        ig_info = ig_info_map.get(out_idx, {})
        # override patient_id from sample_info.txt if available
        if ig_info.get('patient_id'):
            patient_id = ig_info['patient_id']
        if ig_info.get('slice_idx'):
            slice_idx = ig_info['slice_idx']
        if ig_info.get('malignancy_score'):
            malignancy_score = ig_info['malignancy_score']

        # 1. CT grayscale
        save_grayscale(img, sample_out_dir / 'ct.png')

        # 2. Prediction heatmap (inferno)
        save_colormapped(pred, 'inferno', sample_out_dir / 'prediction.png', 0, 1)

        # 3. Prediction overlay (green)
        pred_binary = (pred > 0.4).astype(np.float32)
        save_overlay(img, pred_binary, (0.18, 0.95, 0.32), sample_out_dir / 'pred_overlay.png')

        # 4. GT overlay (blue)
        save_overlay(img, gt.astype(np.float32), (0.22, 0.55, 1.0), sample_out_dir / 'gt_overlay.png')

        # 5. IG images
        has_ig = ig_arr is not None
        has_zoomed_ig = False
        if has_ig:
            save_ig_heatmap(ig_arr, sample_out_dir / 'ig_heatmap.png')
            save_ig_overlay(img, ig_arr, sample_out_dir / 'ig_overlay.png')
            
            # Check for zoomed IG overlay from ig_index0-9
            ig_index_dir = output_dir.parent / 'ig_index0-9'
            if ig_index_dir.exists():
                # Find folder matching index
                for folder in ig_index_dir.iterdir():
                    if folder.is_dir() and folder.name.startswith(f"index_{test_idx}_"):
                        zoomed_src = folder / "3_ig_overlay_zoomed.png"
                        if zoomed_src.exists():
                            import shutil
                            shutil.copy2(zoomed_src, sample_out_dir / 'ig_overlay_zoomed.png')
                            has_zoomed_ig = True
                        break

        metrics = compute_metrics(pred, gt)
        ig_tag = ' [IG]' if has_ig else ''
        print(f"class={class_names_map.get(lbl,'?')}, "
              f"Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}{ig_tag}")

        entry = {
            'id': out_idx,
            'original_index': test_idx,
            'patient_id': str(patient_id),
            'slice_idx': int(slice_idx) if slice_idx is not None else 0,
            'class_label': lbl,
            'class_name': class_names_map.get(lbl, 'Unknown'),
            'malignancy_score': int(malignancy_score) if malignancy_score is not None else 0,
            'has_nodule': has_nodule,
            'metrics': metrics,
            'has_ig': has_ig,
            'ig_stats': {
                'ig_min':  round(float(ig_arr.min()), 8) if has_ig else None,
                'ig_max':  round(float(ig_arr.max()), 8) if has_ig else None,
                'ig_mean': round(float(ig_arr.mean()), 8) if has_ig else None,
                'ig_std':  round(float(ig_arr.std()),  8) if has_ig else None,
            } if has_ig else {},
            'images': {
                'ct':           f'images/sample_{out_idx:02d}/ct.png',
                'prediction':   f'images/sample_{out_idx:02d}/prediction.png',
                'pred_overlay': f'images/sample_{out_idx:02d}/pred_overlay.png',
                'gt_overlay':   f'images/sample_{out_idx:02d}/gt_overlay.png',
                **({'ig_heatmap': f'images/sample_{out_idx:02d}/ig_heatmap.png',
                    'ig_overlay': f'images/sample_{out_idx:02d}/ig_overlay.png',
                    'ig_overlay_zoomed': f'images/sample_{out_idx:02d}/ig_overlay_zoomed.png' if has_zoomed_ig else f'images/sample_{out_idx:02d}/ig_overlay.png'} if has_ig else {})
            }
        }
        samples_manifest.append(entry)

    # ── 6. Aggregate stats ───────────────────────────────────────────────
    all_dice = [s['metrics']['dice'] for s in samples_manifest]
    all_iou  = [s['metrics']['iou']  for s in samples_manifest]

    manifest = {
        'project': 'XAI U-Net: Explainable Lung Nodule Segmentation',
        'description': (
            'Interactive demo of Attention U-Net with Integrated Gradients '
            'for lung nodule segmentation on LIDC-IDRI CT scans.'
        ),
        'dataset': 'LIDC-IDRI',
        'model': 'Attention U-Net',
        'xai_method': 'Integrated Gradients',
        'image_size': list(X_test.shape[1:3]),
        'num_patients_trained': 100,
        'ig_steps': 50,
        'segmentation_threshold': 0.4,
        'total_samples': len(samples_manifest),
        'aggregate_metrics': {
            'mean_dice': round(float(np.mean(all_dice)), 4),
            'mean_iou':  round(float(np.mean(all_iou)),  4),
            'std_dice':  round(float(np.std(all_dice)),  4),
            'std_iou':   round(float(np.std(all_iou)),   4),
        },
        'class_distribution': {
            'Normal':    sum(1 for s in samples_manifest if s['class_label'] == 0),
            'Benign':    sum(1 for s in samples_manifest if s['class_label'] == 1),
            'Malignant': sum(1 for s in samples_manifest if s['class_label'] == 2),
        },
        'samples': samples_manifest,
    }

    manifest_path = dest_dir / 'samples.json'
    with open(str(manifest_path), 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── 7. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output:       {dest_dir.resolve()}")
    print(f"Samples:      {len(samples_manifest)}")
    print(f"With IG:      {sum(1 for s in samples_manifest if s['has_ig'])}")
    print(f"Mean Dice:    {manifest['aggregate_metrics']['mean_dice']:.4f}")
    print(f"Mean IoU:     {manifest['aggregate_metrics']['mean_iou']:.4f}")
    print(f"Classes:      {manifest['class_distribution']}")
    print(f"\nNext: open http://localhost:8080 (or restart the server)")


if __name__ == '__main__':
    main()
