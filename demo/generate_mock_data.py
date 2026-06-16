"""
Generate mock demo data for local frontend testing.
Creates placeholder images and a samples.json manifest
so the frontend can be previewed without real model data.

Usage: python generate_mock_data.py
"""

import os
import json
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Install with: pip install Pillow")
    exit(1)


def make_gradient_image(w, h, color1, color2, noise=True):
    """Create a gradient image with optional noise"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        for c in range(3):
            img[y, :, c] = int(color1[c] * (1 - t) + color2[c] * t)
    if noise:
        noise_arr = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
    return img


def make_ct_mock(w, h):
    """Create a mock CT-like grayscale image"""
    # Dark background with a brighter circular region (like a lung cross-section)
    img = np.random.randint(10, 40, (h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]

    # Two lung-like ellipses
    for offset in [-80, 80]:
        dist = ((X - (cx + offset)) / 100) ** 2 + ((Y - cy) / 120) ** 2
        lung_mask = dist < 1
        img[lung_mask] = np.random.randint(20, 60, np.sum(lung_mask), dtype=np.uint8)

    # Add a small bright nodule
    nx, ny = cx + np.random.randint(-60, 60), cy + np.random.randint(-40, 40)
    nodule_dist = (X - nx) ** 2 + (Y - ny) ** 2
    nodule_mask = nodule_dist < 200
    img[nodule_mask] = np.random.randint(120, 180, np.sum(nodule_mask), dtype=np.uint8)

    return img


def make_heatmap(w, h, cmap_name='inferno'):
    """Create a mock heatmap image"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.random.rand(h, w) * 0.1
    # Add a hotspot
    cy, cx = h // 2 + np.random.randint(-50, 50), w // 2 + np.random.randint(-50, 50)
    Y, X = np.ogrid[:h, :w]
    dist = ((X - cx) ** 2 + (Y - cy) ** 2) / (30 ** 2)
    data += np.exp(-dist) * 0.9

    cmap = plt.cm.get_cmap(cmap_name)
    colored = (cmap(data / data.max())[:, :, :3] * 255).astype(np.uint8)
    return colored


def make_overlay(base_gray, mask, color_rgb, alpha=0.45):
    """Overlay a colored mask on a grayscale image"""
    base_rgb = np.stack([base_gray] * 3, axis=-1).astype(np.float32) / 255
    color_layer = np.zeros((*base_gray.shape, 3), dtype=np.float32)
    for c in range(3):
        color_layer[:, :, c] = color_rgb[c]
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 0:
        mask_f = mask_f / mask_f.max()
    mask_alpha = mask_f[:, :, np.newaxis] * alpha
    blended = base_rgb * (1 - mask_alpha) + color_layer * mask_alpha
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def main():
    output_dir = Path('./data')
    images_dir = output_dir / 'images'

    classes = ['Normal', 'Benign', 'Malignant']
    class_labels = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
    num_samples = 12  # 4 per class

    samples = []

    for i in range(num_samples):
        class_name = classes[i % 3]
        sample_dir = images_dir / f'sample_{i:02d}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        w, h = 256, 256  # Smaller for mock data
        np.random.seed(42 + i)

        # Generate mock images
        ct = make_ct_mock(w, h)
        Image.fromarray(ct, mode='L').save(sample_dir / 'ct.png')

        pred_img = make_heatmap(w, h, 'inferno')
        Image.fromarray(pred_img).save(sample_dir / 'prediction.png')

        # Mock masks
        cy, cx = h // 2 + np.random.randint(-30, 30), w // 2 + np.random.randint(-30, 30)
        Y, X = np.ogrid[:h, :w]
        nodule_mask = ((X - cx) ** 2 + (Y - cy) ** 2) < (15 ** 2)

        pred_overlay = make_overlay(ct, nodule_mask.astype(np.uint8), (0.18, 0.95, 0.32))
        Image.fromarray(pred_overlay).save(sample_dir / 'pred_overlay.png')

        gt_overlay = make_overlay(ct, nodule_mask.astype(np.uint8), (0.22, 0.55, 1.0))
        Image.fromarray(gt_overlay).save(sample_dir / 'gt_overlay.png')

        ig_img = make_heatmap(w, h, 'RdBu_r')
        Image.fromarray(ig_img).save(sample_dir / 'ig_heatmap.png')

        ig_mask = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (25 ** 2)).astype(np.float32)
        ig_overlay = make_overlay(ct, (ig_mask > 0.3).astype(np.uint8), (0.0, 0.95, 0.85), 0.55)
        Image.fromarray(ig_overlay).save(sample_dir / 'ig_overlay.png')

        # Mock metrics
        dice = round(np.random.uniform(0.3, 0.95), 4)
        iou = round(dice * np.random.uniform(0.7, 0.95), 4)

        sample_entry = {
            'id': i,
            'original_index': i * 10,
            'patient_id': 100 + i * 7,
            'slice_idx': np.random.randint(50, 200),
            'class_label': class_labels[class_name],
            'class_name': class_name,
            'malignancy_score': 0 if class_name == 'Normal' else (2 if class_name == 'Benign' else 4),
            'has_nodule': class_name != 'Normal',
            'metrics': {
                'dice': dice,
                'iou': iou,
                'precision': round(np.random.uniform(0.4, 0.98), 4),
                'recall': round(np.random.uniform(0.3, 0.95), 4),
                'gt_area': int(np.random.randint(0, 5000)),
                'pred_area': int(np.random.randint(0, 6000)),
                'intersection': int(np.random.randint(0, 3000)),
                'max_confidence': round(np.random.uniform(0.5, 0.99), 4),
                'mean_confidence': round(np.random.uniform(0.01, 0.1), 6),
            },
            'has_ig': True,
            'ig_stats': {
                'ig_min': round(-np.random.uniform(0.001, 0.01), 6),
                'ig_max': round(np.random.uniform(0.001, 0.01), 6),
                'ig_mean': round(np.random.uniform(-0.0001, 0.0001), 6),
                'ig_std': round(np.random.uniform(0.0001, 0.001), 6),
            },
            'images': {
                'ct': f'images/sample_{i:02d}/ct.png',
                'prediction': f'images/sample_{i:02d}/prediction.png',
                'pred_overlay': f'images/sample_{i:02d}/pred_overlay.png',
                'gt_overlay': f'images/sample_{i:02d}/gt_overlay.png',
                'ig_heatmap': f'images/sample_{i:02d}/ig_heatmap.png',
                'ig_overlay': f'images/sample_{i:02d}/ig_overlay.png',
            }
        }
        samples.append(sample_entry)
        print(f'  Created sample {i:02d}: {class_name}')

    manifest = {
        'project': 'XAI U-Net: Explainable Lung Nodule Segmentation',
        'description': 'Interactive demo (MOCK DATA for preview)',
        'dataset': 'LIDC-IDRI',
        'model': 'Attention U-Net',
        'xai_method': 'Integrated Gradients',
        'image_size': [512, 512],
        'num_patients_trained': 100,
        'ig_steps': 50,
        'segmentation_threshold': 0.4,
        'total_samples': len(samples),
        'aggregate_metrics': {
            'mean_dice': 0.6543,
            'mean_iou': 0.5234,
            'std_dice': 0.1876,
            'std_iou': 0.1543,
        },
        'class_distribution': {
            'Normal': sum(1 for s in samples if s['class_label'] == 0),
            'Benign': sum(1 for s in samples if s['class_label'] == 1),
            'Malignant': sum(1 for s in samples if s['class_label'] == 2),
        },
        'samples': samples,
    }

    with open(output_dir / 'samples.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nMock data generated: {len(samples)} samples')
    print(f'Output: {output_dir.resolve()}')
    print(f'\nStart the demo with:')
    print(f'  python -m http.server 8080')
    print(f'  Then open http://localhost:8080')


if __name__ == '__main__':
    main()
