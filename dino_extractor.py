"""
DINO Feature Extractor for KalmanTrack
Extracts semantic features using DINO vision transformer for object tracking
"""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")


class DINOFeatureExtractor:
    """
    DINO-based feature extractor that converts semantic features to trackable keypoints
    """
    
    def __init__(self, model_name: str = 'dino_vits16', n_keypoints: int = 50, device: str = 'auto'):
        """
        Initialize DINO feature extractor
        
        Args:
            model_name: DINO model variant ('dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8')
            n_keypoints: Number of keypoints to extract
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.n_keypoints = n_keypoints
        self.patch_size = 16 if 's16' in model_name or 'b16' in model_name else 8
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading DINO model {model_name} on {self.device}...")
        
        # Load pre-trained DINO model
        try:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading DINO model: {e}")
            print("Falling back to local model if available...")
            raise
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Store reference features for matching
        self.reference_features = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.reference_roi = None  # Store ROI for focused tracking
        self.last_point_ref_indices = []  # Track which reference points are matched
        
        # Multi-scale feature caching (disabled by default)
        self.use_multiscale = False  # Disabled - caused worse performance
        self.scales = [1.0]  # Single scale only
        self.feature_cache = {}  # Cache features to avoid recomputation
        
    def extract_patch_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract patch-level features from image using DINO
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Patch features tensor of shape (num_patches, feature_dim)
        """
        # Convert to PIL and preprocess
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get intermediate layer features (patch-level)
            features = self.model.get_intermediate_layers(input_tensor, n=1)[0]
            # Remove CLS token, keep only patch tokens
            patch_features = features[:, 1:, :]  # Shape: (1, num_patches, feature_dim)
            
        return patch_features.squeeze(0)  # Shape: (num_patches, feature_dim)
    
    def features_to_keypoints(self, patch_features: torch.Tensor, 
                            image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert patch features to keypoints using clustering
        
        Args:
            patch_features: Patch features tensor (num_patches, feature_dim)
            image_shape: Original image shape (height, width)
            
        Returns:
            keypoints: Array of keypoint coordinates (n_keypoints, 2)
            descriptors: Array of feature descriptors (n_keypoints, feature_dim)
        """
        features_np = patch_features.cpu().numpy()
        
        # Use K-means clustering to find representative patches
        n_clusters = min(self.n_keypoints, features_np.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_np)
        cluster_centers = kmeans.cluster_centers_
        
        # Calculate patches per side (assuming square patch grid)
        num_patches = features_np.shape[0]
        patches_per_side = int(np.sqrt(num_patches))
        
        # Scale factors to map from 224x224 to original image size
        scale_y = image_shape[0] / 224.0
        scale_x = image_shape[1] / 224.0
        
        keypoints = []
        descriptors = []
        
        for i, center in enumerate(cluster_centers):
            # Find the patch closest to this cluster center
            distances = np.linalg.norm(features_np - center, axis=1)
            closest_patch_idx = np.argmin(distances)
            
            # Convert patch index to image coordinates
            row = closest_patch_idx // patches_per_side
            col = closest_patch_idx % patches_per_side
            
            # Calculate center of patch in 224x224 space
            patch_center_x = col * self.patch_size + self.patch_size // 2
            patch_center_y = row * self.patch_size + self.patch_size // 2
            
            # Scale to original image size
            x = patch_center_x * scale_x
            y = patch_center_y * scale_y
            
            keypoints.append([x, y])
            descriptors.append(features_np[closest_patch_idx])
        
        return np.array(keypoints), np.array(descriptors)
    
    def _extract_corner_based_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract DINO features at corner locations (like FAST algorithm)
        
        Args:
            image: Input image
            
        Returns:
            keypoints: Corner locations
            descriptors: DINO features at those locations
        """
        # Convert to grayscale for corner detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect corners using goodFeaturesToTrack (optimized for stability)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.n_keypoints * 2,  # Get more candidates
            qualityLevel=0.01,   # Higher quality threshold
            minDistance=5,       # Allow closer corners
            blockSize=3,         # Smaller block for precision
            useHarrisDetector=False,  # Use Shi-Tomasi (more stable)
        )
        
        if corners is None or len(corners) == 0:
            return np.array([]), np.array([])
        
        # Convert corner format
        keypoints = corners.reshape(-1, 2)
        
        # Extract DINO features at corner locations
        descriptors = self._extract_dino_at_points(image, keypoints)
        
        return keypoints, descriptors
    
    def _extract_dino_at_points(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale DINO features at specific point locations
        
        Args:
            image: Input image
            points: Point coordinates (N, 2)
            
        Returns:
            Multi-scale feature descriptors at those points
        """
        if not self.use_multiscale:
            # Single scale (original implementation)
            patch_features = self.extract_patch_features(image)
            return self._extract_features_at_points_single_scale(image, points, patch_features)
        
        # Multi-scale feature extraction
        all_descriptors = []
        
        for scale in self.scales:
            # Resize image for this scale
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h))
                scaled_points = points * scale
            else:
                scaled_image = image
                scaled_points = points
            
            # Extract DINO features for this scale
            patch_features = self.extract_patch_features(scaled_image)
            scale_descriptors = self._extract_features_at_points_single_scale(
                scaled_image, scaled_points, patch_features
            )
            
            all_descriptors.append(scale_descriptors)
        
        # Concatenate multi-scale features
        if len(all_descriptors) > 1:
            # Concatenate features from different scales
            multi_scale_descriptors = np.concatenate(all_descriptors, axis=1)
        else:
            multi_scale_descriptors = all_descriptors[0]
        
        return multi_scale_descriptors
    
    def _extract_features_at_points_single_scale(self, image: np.ndarray, points: np.ndarray, 
                                               patch_features: torch.Tensor) -> np.ndarray:
        """
        Extract DINO features at points for a single scale
        
        Args:
            image: Input image
            points: Point coordinates (N, 2)
            patch_features: Pre-computed patch features
            
        Returns:
            Feature descriptors at those points
        """
        # Map points to patch indices
        patch_size = self.patch_size
        patches_per_side = int(np.sqrt(patch_features.shape[0]))
        
        # Scale factors from image to 224x224 DINO input
        scale_y = 224.0 / image.shape[0]
        scale_x = 224.0 / image.shape[1]
        
        descriptors = []
        for point in points:
            # Convert point to DINO patch coordinates
            scaled_x = point[0] * scale_x
            scaled_y = point[1] * scale_y
            
            # Find nearest patch
            patch_x = int(scaled_x // patch_size)
            patch_y = int(scaled_y // patch_size)
            
            # Clamp to valid range
            patch_x = max(0, min(patch_x, patches_per_side - 1))
            patch_y = max(0, min(patch_y, patches_per_side - 1))
            
            # Get patch index
            patch_idx = patch_y * patches_per_side + patch_x
            
            if patch_idx < patch_features.shape[0]:
                descriptors.append(patch_features[patch_idx].cpu().numpy())
            else:
                # Fallback to last patch if out of bounds
                descriptors.append(patch_features[-1].cpu().numpy())
        
        return np.array(descriptors)
    
    def _compute_adaptive_threshold(self, similarity_matrix: np.ndarray, base_threshold: float = 0.3) -> float:
        """
        Compute adaptive threshold based on similarity distribution
        
        Args:
            similarity_matrix: Matrix of similarity scores
            base_threshold: Base threshold to adjust from
            
        Returns:
            Adaptive threshold value
        """
        # Get statistics of similarity scores
        max_similarities = np.max(similarity_matrix, axis=1)
        mean_max_sim = np.mean(max_similarities)
        std_max_sim = np.std(max_similarities)
        
        # Adaptive threshold based on distribution
        # If matches are generally good (high mean), lower threshold
        # If matches are poor (low mean), raise threshold to be more selective
        if mean_max_sim > 0.6:  # Good matches available
            adaptive_threshold = max(base_threshold - 0.1, 0.2)
        elif mean_max_sim < 0.4:  # Poor matches, be more selective
            adaptive_threshold = min(base_threshold + 0.1, 0.5)
        else:
            # Use percentile-based approach for moderate cases
            adaptive_threshold = np.percentile(max_similarities, 70)  # 70th percentile
            adaptive_threshold = max(min(adaptive_threshold, 0.5), 0.2)  # Clamp to reasonable range
        
        return adaptive_threshold
    
    def extract_keypoints_and_descriptors(self, image: np.ndarray, use_corners: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoints and descriptors from image using corner detection + DINO features
        
        Args:
            image: Input image as numpy array
            use_corners: If True, use corner detection for keypoints; if False, use clustering
            
        Returns:
            keypoints: Array of keypoint coordinates (n_keypoints, 2)
            descriptors: Array of feature descriptors (n_keypoints, feature_dim)
        """
        if use_corners:
            return self._extract_corner_based_features(image)
        else:
            # Fallback to original clustering method
            patch_features = self.extract_patch_features(image)
            keypoints, descriptors = self.features_to_keypoints(patch_features, image.shape[:2])
            return keypoints, descriptors
    
    def set_reference(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Set reference image/ROI for tracking
        
        Args:
            image: Reference image
            roi: Region of interest (x_min, y_min, x_max, y_max)
        """
        self.reference_roi = roi
        
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            roi_image = image[y_min:y_max, x_min:x_max]
        else:
            roi_image = image
            
        # Use corner-based features for better tracking stability
        self.reference_keypoints, self.reference_descriptors = self.extract_keypoints_and_descriptors(roi_image, use_corners=True)
        print(f"Reference set with {len(self.reference_keypoints)} corner-based keypoints from ROI {roi} (will search full frame for matches)")
    
    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray, 
                      threshold: float = 0.3, adaptive: bool = True) -> List[Tuple[int, int]]:
        """
        Match features between two descriptor sets using cosine similarity
        
        Args:
            descriptors1: First set of descriptors
            descriptors2: Second set of descriptors
            threshold: Matching threshold
            
        Returns:
            List of matches as (idx1, idx2) tuples
        """
        # Normalize descriptors for cosine similarity
        desc1_norm = descriptors1 / (np.linalg.norm(descriptors1, axis=1, keepdims=True) + 1e-8)
        desc2_norm = descriptors2 / (np.linalg.norm(descriptors2, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(desc1_norm, desc2_norm.T)
        
        # Adaptive thresholding based on similarity distribution
        if adaptive:
            threshold = self._compute_adaptive_threshold(similarity_matrix, base_threshold=threshold)
        
        # Debug: print similarity statistics (disabled for cleaner output)
        # max_sim = np.max(similarity_matrix)
        # mean_sim = np.mean(similarity_matrix)
        # print(f"Similarity stats: max={max_sim:.3f}, mean={mean_sim:.3f}, threshold={threshold}")
        
        matches = []
        
        # Simple threshold-based matching (more robust)
        for i in range(len(descriptors1)):
            similarities = similarity_matrix[i]
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]
            
            # Simple threshold test
            if best_sim > threshold:
                matches.append((i, best_idx))
        
        # Remove duplicate matches (keep best scoring)
        unique_matches = {}
        for ref_idx, curr_idx in matches:
            sim_score = similarity_matrix[ref_idx, curr_idx]
            if curr_idx not in unique_matches or sim_score > unique_matches[curr_idx][1]:
                unique_matches[curr_idx] = (ref_idx, sim_score)
        
        # Convert back to list
        final_matches = [(ref_idx, curr_idx) for curr_idx, (ref_idx, _) in unique_matches.items()]
        
        return final_matches
    
    def track_points(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Track multiple points in current frame (searches entire frame like FAST algorithm)
        
        Args:
            image: Current frame
            
        Returns:
            List of tracked point coordinates [(x, y), ...] or None if tracking failed
        """
        if self.reference_descriptors is None:
            print("No reference set. Call set_reference() first.")
            return None
        
        # Extract corner-based features from ENTIRE frame (like FAST algorithm approach)
        # This allows tracking the object as it moves anywhere in the frame
        current_keypoints, current_descriptors = self.extract_keypoints_and_descriptors(image, use_corners=True)
        
        # Match with reference features (disable adaptive for now)
        matches = self.match_features(self.reference_descriptors, current_descriptors, adaptive=False)
        
        # Debug info (disabled for cleaner output)
        # print(f"Feature matching: {len(matches)} matches found (ref: {len(self.reference_descriptors)}, curr: {len(current_descriptors)})")
        # if len(matches) > 0:
        #     print(f"Sample matches: {matches[:3]}")
        
        if len(matches) == 0:  # Back to 3 minimum matches
            print(f"Insufficient matches: {len(matches)}/3 required")
            return None
        
        # Return all matched keypoints with reference indices for consistent coloring
        tracked_points = []
        point_ref_indices = []  # Store which reference point each tracked point corresponds to
        
        for ref_idx, curr_idx in matches:
            x, y = current_keypoints[curr_idx]
            tracked_points.append((int(x), int(y)))
            point_ref_indices.append(ref_idx)
        
        print(f"Tracked {len(tracked_points)} points in full frame (following moving object)")
        
        # Store reference indices for consistent coloring
        self.last_point_ref_indices = point_ref_indices
        return tracked_points
    
    def track_object(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Track object center (backward compatibility)
        
        Args:
            image: Current frame
            
        Returns:
            Center coordinates (x, y) or None if tracking failed
        """
        points = self.track_points(image)
        if points is None or len(points) == 0:
            return None
        
        # Calculate center from all tracked points
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        return int(center_x), int(center_y)
    
    def visualize_keypoints(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Visualize keypoints on image
        
        Args:
            image: Input image
            keypoints: Keypoints to visualize
            
        Returns:
            Image with keypoints drawn
        """
        vis_image = image.copy()
        for kp in keypoints:
            cv2.circle(vis_image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
        return vis_image
