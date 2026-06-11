#!/usr/bin/env python3
"""
Check if the model predictions are reproducible and consistent
"""

import numpy as np
import tensorflow as tf
import os

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_model():
    """Load the model with custom objects"""
    
    def focal_tversky_loss(y_tru