"""Load and validate mask images from dataset."""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2


class MaskLoader:
    """Handle loading and validation of mask images."""
    
    def __init__(self, masks_dir: str):
        """Initialize mask loader.
        
        Args:
            masks_dir: Directory containing mask PNG files
        """
        self.masks_dir = Path(masks_dir)
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory does not exist: {masks_dir}")
        
        self.mask_files = sorted(list(self.masks_dir.glob("*.png")))
        if not self.mask_files:
            raise ValueError(f"No PNG files found in: {masks_dir}")
        
        print(f"Found {len(self.mask_files)} masks in {masks_dir}")
    
    def load_mask(self, filepath: Path) -> np.ndarray:
        """Load a single mask from file.
        
        Args:
            filepath: Path to mask PNG file
            
        Returns:
            Binary mask as numpy array (values 0 or 255)
        """
        mask = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {filepath}")
        
        # Ensure binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask
    
    def load_random_masks(self, count: int, seed: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """Load random masks from dataset.
        
        Args:
            count: Number of masks to load
            seed: Random seed for reproducibility
            
        Returns:
            List of tuples (mask_id, mask_array)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if count > len(self.mask_files):
            print(f"Warning: Requested {count} masks but only {len(self.mask_files)} available.")
            print(f"Will sample with replacement.")
        
        selected_files = np.random.choice(self.mask_files, size=count, replace=True)
        
        masks = []
        for filepath in selected_files:
            mask = self.load_mask(filepath)
            mask_id = filepath.stem
            masks.append((mask_id, mask))
        
        return masks
    
    def get_all_masks(self) -> List[Tuple[str, np.ndarray]]:
        """Load all masks from dataset.
        
        Returns:
            List of tuples (mask_id, mask_array)
        """
        masks = []
        for filepath in self.mask_files:
            mask = self.load_mask(filepath)
            mask_id = filepath.stem
            masks.append((mask_id, mask))
        
        return masks
    
    def get_mask_shape(self) -> Tuple[int, int]:
        """Get the shape of masks in dataset.
        
        Returns:
            Tuple of (height, width)
        """
        if not self.mask_files:
            return (0, 0)
        
        first_mask = self.load_mask(self.mask_files[0])
        return first_mask.shape
    
    def validate_masks(self) -> bool:
        """Validate that all masks have consistent dimensions.
        
        Returns:
            True if all masks are valid and consistent
        """
        if not self.mask_files:
            return False
        
        reference_shape = None
        for filepath in self.mask_files:
            try:
                mask = self.load_mask(filepath)
                if reference_shape is None:
                    reference_shape = mask.shape
                elif mask.shape != reference_shape:
                    print(f"Warning: Inconsistent shape in {filepath.name}: "
                          f"{mask.shape} vs {reference_shape}")
                    return False
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")
                return False
        
        print(f"All masks validated successfully. Shape: {reference_shape}")
        return True
