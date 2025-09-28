#!/usr/bin/env python3
"""
Debug script to test DINO feature extraction and matching
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dino_extractor import DINOFeatureExtractor

def debug_feature_extraction():
    """Debug DINO feature extraction on a single frame"""
    
    # Load video and get first frame
    video_path = 'videos/helicopter.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not load frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Initialize extractor
    extractor = DINOFeatureExtractor(
        model_name='dino_vits16',
        n_keypoints=50,
        device='auto'
    )
    
    # Test different ROI sizes and positions
    height, width = frame.shape[:2]
    
    test_rois = [
        (width//4, height//4, 3*width//4, 3*height//4, "Large Center"),
        (width//2-50, height//2-50, width//2+50, height//2+50, "Small Center"),
        (50, 50, 200, 200, "Top-Left Corner"),
        (width-200, height-200, width-50, height-50, "Bottom-Right Corner"),
    ]
    
    for x_min, y_min, x_max, y_max, name in test_rois:
        print(f"\nTesting ROI: {name} - ({x_min}, {y_min}, {x_max}, {y_max})")
        
        # Extract ROI
        roi_image = frame[y_min:y_max, x_min:x_max]
        print(f"ROI shape: {roi_image.shape}")
        
        # Extract features
        try:
            keypoints, descriptors = extractor.extract_keypoints_and_descriptors(roi_image)
            print(f"Extracted {len(keypoints)} keypoints, descriptor shape: {descriptors.shape}")
            
            # Visualize keypoints
            vis_roi = extractor.visualize_keypoints(roi_image, keypoints)
            
            # Show ROI with keypoints
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
            plt.title(f'{name} - Original ROI')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(vis_roi, cv2.COLOR_BGR2RGB))
            plt.title(f'{name} - With Keypoints')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'debug_roi_{name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()


def debug_feature_matching():
    """Debug feature matching between two similar frames"""
    
    # Load video
    video_path = 'videos/helicopter.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Get two consecutive frames
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()
    
    if not ret1 or not ret2:
        print("Could not load frames")
        return
    
    # Initialize extractor
    extractor = DINOFeatureExtractor(
        model_name='dino_vits16',
        n_keypoints=50,
        device='auto'
    )
    
    # Use center ROI
    height, width = frame1.shape[:2]
    roi = (width//2-100, height//2-100, width//2+100, height//2+100)
    x_min, y_min, x_max, y_max = roi
    
    roi1 = frame1[y_min:y_max, x_min:x_max]
    roi2 = frame2[y_min:y_max, x_min:x_max]
    
    print(f"ROI shape: {roi1.shape}")
    
    # Extract features from both frames
    kp1, desc1 = extractor.extract_keypoints_and_descriptors(roi1)
    kp2, desc2 = extractor.extract_keypoints_and_descriptors(roi2)
    
    print(f"Frame 1: {len(kp1)} keypoints")
    print(f"Frame 2: {len(kp2)} keypoints")
    
    # Test matching with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("\nMatching results:")
    for threshold in thresholds:
        matches = extractor.match_features(desc1, desc2, threshold=threshold)
        print(f"Threshold {threshold}: {len(matches)} matches")
    
    # Visualize best matching result
    best_threshold = 0.4
    matches = extractor.match_features(desc1, desc2, threshold=best_threshold)
    
    print(f"\nUsing threshold {best_threshold}: {len(matches)} matches")
    
    # Create visualization
    vis1 = extractor.visualize_keypoints(roi1, kp1)
    vis2 = extractor.visualize_keypoints(roi2, kp2)
    
    # Draw matches
    if len(matches) > 0:
        match_vis = np.hstack([vis1, vis2])
        
        for ref_idx, curr_idx in matches[:10]:  # Show first 10 matches
            pt1 = (int(kp1[ref_idx][0]), int(kp1[ref_idx][1]))
            pt2 = (int(kp2[curr_idx][0] + vis1.shape[1]), int(kp2[curr_idx][1]))
            
            cv2.line(match_vis, pt1, pt2, (0, 255, 255), 1)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches (threshold={best_threshold}, matches={len(matches)})')
        plt.axis('off')
        plt.savefig('debug_feature_matches.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return len(matches)


def test_tracking_on_static_frame():
    """Test tracking on the same frame (should have perfect matches)"""
    
    # Load frame
    video_path = 'videos/helicopter.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not load frame")
        return
    
    # Initialize extractor
    extractor = DINOFeatureExtractor(
        model_name='dino_vits16',
        n_keypoints=50,
        device='auto'
    )
    
    # Set reference
    height, width = frame.shape[:2]
    roi = (width//2-100, height//2-100, width//2+100, height//2+100)
    
    extractor.set_reference(frame, roi)
    
    # Track on same frame (should work perfectly)
    result = extractor.track_object(frame)
    
    print(f"Self-tracking result: {result}")
    
    if result is not None:
        print("✓ Self-tracking successful - DINO extraction is working")
        return True
    else:
        print("✗ Self-tracking failed - Problem with DINO extraction")
        return False


if __name__ == "__main__":
    print("DINO Feature Debug Script")
    print("=" * 40)
    
    print("\n1. Testing self-tracking (same frame)...")
    if test_tracking_on_static_frame():
        print("\n2. Testing feature extraction on different ROIs...")
        debug_feature_extraction()
        
        print("\n3. Testing feature matching between frames...")
        matches = debug_feature_matching()
        
        if matches > 0:
            print(f"\n✓ Feature matching is working ({matches} matches found)")
        else:
            print("\n✗ Feature matching is not working")
    else:
        print("\n✗ Basic DINO extraction is not working")
    
    print("\nDebug complete. Check generated images for visual results.")
