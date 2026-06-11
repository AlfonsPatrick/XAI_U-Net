"""
Test script to check if all imports and files are working correctly
"""

import os
import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True


def test_file_existence():
    """Test if required files exist"""
    print("\nTesting file existence...")
    
    required_files = [
        'integrated_gradients_unet.py',
        'example_ig_usage.py',
        'standalone_ig_example.py'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"❌ {file} not found")
            all_exist = False
    
    return all_exist


def test_module_import():
    """Test importing the integrated gradients module"""
    print("\nTesting module import...")
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    try:
        from integrated_gradients_unet import IntegratedGradientsExplainer, Config
        print("✓ integrated_gradients_unet module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return False


def test_data_files():
    """Test if data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        'workspace/output_latest/comprehensive_results.npz',
        'workspace/output_latest/detailed_metadata.pkl'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"⚠️  {file} not found (needed for analysis)")


def main():
    """Run all tests"""
    print("="*60)
    print("INTEGRATED GRADIENTS SETUP TEST")
    print("="*60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test files
    files_ok = test_file_existence()
    
    # Test module import
    module_ok = test_module_import()
    
    # Test data files
    test_data_files()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if imports_ok and files_ok and module_ok:
        print("✅ All tests passed! You can run the integrated gradients analysis.")
        print("\nTo run the analysis, use one of these options:")
        print("1. python example_ig_usage.py")
        print("2. python standalone_ig_example.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        
        if not imports_ok:
            print("- Install missing Python packages")
        if not files_ok:
            print("- Ensure all required files are in the current directory")
        if not module_ok:
            print("- Check that integrated_gradients_unet.py is valid Python code")


if __name__ == "__main__":
    main()