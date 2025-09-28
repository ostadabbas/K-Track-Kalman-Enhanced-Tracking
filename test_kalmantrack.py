#!/usr/bin/env python3
"""
Test script for KalmanTrack installation and basic functionality
"""

import sys
import os
import numpy as np
import cv2

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True


def test_dino_model():
    """Test DINO model loading"""
    print("\nTesting DINO model loading...")
    
    try:
        import torch
        
        # Try to load DINO model
        print("Loading DINO model (this may take a while on first run)...")
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        print("✓ DINO model loaded successfully")
        
        # Test model inference
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ DINO inference test passed, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ DINO model test failed: {e}")
        return False


def test_kalmantrack_modules():
    """Test KalmanTrack module imports"""
    print("\nTesting KalmanTrack modules...")
    
    try:
        from dino_extractor import DINOFeatureExtractor
        print("✓ DINOFeatureExtractor imported")
    except ImportError as e:
        print(f"✗ DINOFeatureExtractor import failed: {e}")
        return False
    
    try:
        from kalman_filter import KalmanTracker
        print("✓ KalmanTracker imported")
    except ImportError as e:
        print(f"✗ KalmanTracker import failed: {e}")
        return False
    
    try:
        from kalman_track import KalmanTrack
        print("✓ KalmanTrack imported")
    except ImportError as e:
        print(f"✗ KalmanTrack import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with synthetic data"""
    print("\nTesting basic functionality...")
    
    try:
        from kalman_track import KalmanTrack
        
        # Create synthetic image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize tracker (use CPU for testing)
        tracker = KalmanTrack(device='cpu', n_keypoints=10)
        print("✓ KalmanTrack initialized")
        
        # Test ROI setting
        roi = (100, 100, 200, 200)
        tracker.set_target(test_image, roi)
        print("✓ Target set successfully")
        
        # Test tracking on a few frames
        for i in range(3):
            # Create slightly different image
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = tracker.track(test_frame)
            print(f"✓ Frame {i+1} tracking: success={result.success}, time={result.processing_time:.3f}s")
        
        # Test performance stats
        stats = tracker.get_performance_stats()
        print(f"✓ Performance stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("KalmanTrack Installation Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return False
    
    # Test KalmanTrack modules
    if not test_kalmantrack_modules():
        print("\n❌ KalmanTrack module tests failed. Check file paths and syntax.")
        return False
    
    # Test DINO model (optional, may be slow)
    print("\nDINO model test (optional, may be slow)...")
    response = input("Test DINO model loading? This may take a while on first run. (y/N): ")
    if response.lower() in ['y', 'yes']:
        if not test_dino_model():
            print("⚠️  DINO model test failed. You may need internet connection or GPU memory.")
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 All tests passed! KalmanTrack is ready to use.")
    print("\nNext steps:")
    print("1. Run: python demo.py --video videos/your_video.mp4")
    print("2. Select ROI by dragging mouse on the first frame")
    print("3. Watch the tracking in action!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
