"""
Extract Integrated Gradient results as PNG images for samples 0-9.
"""

import os
import sys
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    print("Please install Pillow and matplotlib")
    sys.exit(1)

def save_grayscale(arr_2d, path):
    arr = np.squeeze(arr_2d).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if abs(hi - lo) < 1e-9:
        img = np.zeros_like(arr, dtype=np.uint8)
    else:
        img = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(path, optimize=True)

def save_colormapped(arr_2d, cmap_name, path, vmin=None, vmax=None):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    arr = np.squeeze(arr_2d).astype(np.float32)
    _vmin = float(arr.min()) if vmin is None else vmin
    _vmax = float(arr.max()) if vmax is None else vmax
    if abs(_vmax - _vmin) < 1e-9:
        _vmax = _vmin + 1.0
    norm = np.clip((arr - _vmin) / (_vmax - _vmin), 0, 1)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path, optimize=True)

def save_ig_heatmap(ig_2d, path):
    arr = np.abs(np.squeeze(ig_2d).astype(np.float32))
    vmax = float(arr.max())
    if vmax < 1e-12:
        vmax = 1.0
    save_colormapped(arr, 'Blues', path, vmin=0, vmax=vmax)

def save_overlay(base_gray, mask, color_rgb, path, alpha=0.45):
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

def save_ig_overlay(base_gray, ig_2d, path, threshold=0.15):
    mag = np.abs(np.squeeze(ig_2d).astype(np.float32))
    mag_max = mag.max()
    if mag_max > 1e-12:
        mag = mag / mag_max  # Normalize so max magnitude is exactly 1.0
        
    mask = np.clip(mag, 0, 1)
    save_overlay(base_gray, (mask > threshold).astype(np.float32) * mask,
                 (0.0, 0.95, 0.85), path, alpha=0.8)

def main():
    output_dir = Path(r"d:\docs\UGM\KULIYEAH\skripsi fix\output")
    dest_dir = Path(r"d:\docs\UGM\KULIYEAH\skripsi fix\ig_index0-9")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    npz_path = output_dir / 'comprehensive_results.npz'
    print(f"Loading CT data from {npz_path}...")
    data = np.load(str(npz_path))
    X_test = data['X_test']

    print(f"Extracting 10 IG samples to {dest_dir}...")
    for i in range(10):
        ig_dir = output_dir / f"ig_sample_{i}"
        npy_path = ig_dir / "integrated_gradients.npy"
        info_path = ig_dir / "sample_info.txt"
        
        if not npy_path.exists():
            print(f"Skipping {i}, no npy file.")
            continue
            
        ig_arr = np.load(str(npy_path))
        
        # Get patient id from sample_info
        patient_id = f"unknown"
        slice_idx = f"unknown"
        if info_path.exists():
            with open(info_path, 'r') as f:
                for line in f:
                    if 'patient_id' in line:
                        patient_id = line.split(':')[1].strip()
                    if 'slice_idx' in line:
                        slice_idx = line.split(':')[1].strip()
                        
        sample_out_dir = dest_dir / f"index_{i}_patient_{patient_id}_slice_{slice_idx}"
        sample_out_dir.mkdir(exist_ok=True)
        
        ct_img = np.squeeze(X_test[i])
        
        # Save results
        save_grayscale(ct_img, sample_out_dir / "1_original_ct.png")
        save_ig_heatmap(ig_arr, sample_out_dir / "2_ig_heatmap.png")
        save_ig_overlay(ct_img, ig_arr, sample_out_dir / "3_ig_overlay.png")
        
        print(f"Processed Index {i}: Patient {patient_id}, Slice {slice_idx}")

    print("\nDone! Images saved to:", dest_dir)

if __name__ == '__main__':
    main()
