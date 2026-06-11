#!/usr/bin/env python3
"""
Simple Attention U-Net Testing Script

This script provides a streamlined approach to test the attention U-Net model
and visualize attention maps for several test cases.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

def setup_gpu():
    """Setup GPU memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU setup complete. Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")

def find_model_files():
    """Find the model and data files in the workspace"""
    possible_paths = [
        "/workspace/output/",
        "./workspace/output/",
        "./output/",
        "./"
    ]
    
    files_found = {}
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            print(f"Checking directory: {base_path}")
            
            # Look for model file
            model_paths = [
                os.path.join(base_path, "models", "best_unet_model.h5"),
                os.path.join(base_path, "best_unet_model.h5")
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    files_found['model'] = model_path
                    print(f"  ✓ Found model: {model_path}")
                    break
            
            # Look for data file
            data_paths = [
                os.path.join(base_path, "comprehensive_results.npz"),
                os.path.join(base_path, "data_splits.npz")
            ]
            
            for data_path in data_paths:
                if os.path.exists(data_path):
                    files_found['data'] = data_path
                    print(f"  ✓ Found data: {data_path}")
                    break
            
            # Look for metadata file
            metadata_paths = [
                o