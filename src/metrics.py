"""Metrics for evaluating mask quality and diversity."""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from itertools import combinations
from tqdm import tqdm


class MaskMetrics:
    """Calculate metrics for mask evaluation."""
    
    @staticmethod
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Intersection over Union between two masks.
        
        Args:
            mask1: Binary mask 1
            mask2: Binary mask 2
            
        Returns:
            IoU score (0 to 1)
        """
        # Ensure binary
        mask1 = (mask1 > 127).astype(np.uint8)
        mask2 = (mask2 > 127).astype(np.uint8)
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    @staticmethod
    def calculate_pairwise_iou(masks: List[np.ndarray]) -> Dict[str, float]:
        """Calculate pairwise IoU for all mask pairs.
        
        Args:
            masks: List of binary masks
            
        Returns:
            Dictionary with IoU statistics
        """
        n_masks = len(masks)
        print(f"\nCalculating pairwise IoU for {n_masks} masks...")
        
        iou_values = []
        
        # Calculate IoU for all pairs
        for i, j in tqdm(list(combinations(range(n_masks), 2)), 
                         desc="Computing IoU pairs"):
            iou = MaskMetrics.calculate_iou(masks[i], masks[j])
            iou_values.append(iou)
        
        iou_values = np.array(iou_values)
        
        return {
            'mean_iou': float(np.mean(iou_values)),
            'std_iou': float(np.std(iou_values)),
            'min_iou': float(np.min(iou_values)),
            'max_iou': float(np.max(iou_values)),
            'median_iou': float(np.median(iou_values)),
            'n_pairs': len(iou_values)
        }
    
    @staticmethod
    def calculate_shape_descriptors(mask: np.ndarray) -> Dict[str, float]:
        """Calculate shape descriptors for a mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary of shape descriptors
        """
        # Ensure binary
        mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Empty mask
            return {
                'area': 0.0,
                'perimeter': 0.0,
                'compactness': 0.0,
                'eccentricity': 0.0,
                'solidity': 0.0,
                'extent': 0.0
            }
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Area
        area = cv2.contourArea(contour)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Compactness (circularity): 4π*area / perimeter²
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            compactness = 0.0
        
        # Eccentricity (from fitted ellipse)
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            except:
                eccentricity = 0.0
        
        # Solidity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Extent: area / bounding box area
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0.0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'extent': float(extent)
        }
    
    @staticmethod
    def calculate_shape_statistics(masks: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate shape descriptor statistics for a set of masks.
        
        Args:
            masks: List of binary masks
            
        Returns:
            Dictionary of statistics for each descriptor
        """
        print(f"\nCalculating shape descriptors for {len(masks)} masks...")
        
        # Collect descriptors for all masks
        all_descriptors = {
            'area': [],
            'perimeter': [],
            'compactness': [],
            'eccentricity': [],
            'solidity': [],
            'extent': []
        }
        
        for mask in tqdm(masks, desc="Computing shape descriptors"):
            descriptors = MaskMetrics.calculate_shape_descriptors(mask)
            for key, value in descriptors.items():
                all_descriptors[key].append(value)
        
        # Calculate statistics
        statistics = {}
        for key, values in all_descriptors.items():
            values = np.array(values)
            statistics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        return statistics
    
    @staticmethod
    def calculate_diversity_score(mean_iou: float) -> float:
        """Calculate diversity score from mean IoU.
        
        Lower IoU indicates higher diversity.
        
        Args:
            mean_iou: Mean pairwise IoU
            
        Returns:
            Diversity score (0 to 1, higher is more diverse)
        """
        # Diversity score: 1 - mean_iou
        return 1.0 - mean_iou
    
    @staticmethod
    def calculate_coverage(masks: List[np.ndarray]) -> float:
        """Calculate average coverage (non-zero pixels ratio).
        
        Args:
            masks: List of binary masks
            
        Returns:
            Average coverage ratio
        """
        coverages = []
        for mask in masks:
            total_pixels = mask.size
            non_zero_pixels = np.count_nonzero(mask > 127)
            coverage = non_zero_pixels / total_pixels
            coverages.append(coverage)
        
        return float(np.mean(coverages))
