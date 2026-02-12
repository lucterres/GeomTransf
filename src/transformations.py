"""Geometric transformations for mask augmentation."""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple, Optional


class MaskTransformer:
    """Apply geometric transformations to binary masks."""
    
    @staticmethod
    def rotate(mask: np.ndarray, angle: float) -> np.ndarray:
        """Rotate mask by given angle.
        
        Args:
            mask: Binary mask (H, W)
            angle: Rotation angle in degrees (counterclockwise)
            
        Returns:
            Rotated mask
        """
        h, w = mask.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), 
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        return rotated
    
    @staticmethod
    def scale(mask: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale mask by given factor.
        
        Args:
            mask: Binary mask (H, W)
            scale_factor: Scale factor (>1 enlarges, <1 shrinks)
            
        Returns:
            Scaled mask (same size as input, centered)
        """
        h, w = mask.shape
        center = (w // 2, h // 2)
        
        # Create scaling transformation matrix
        scale_matrix = cv2.getRotationMatrix2D(center, 0, scale_factor)
        scaled = cv2.warpAffine(mask, scale_matrix, (w, h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
        return scaled
    
    @staticmethod
    def flip_horizontal(mask: np.ndarray) -> np.ndarray:
        """Flip mask horizontally.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Horizontally flipped mask
        """
        return cv2.flip(mask, 1)
    
    @staticmethod
    def flip_vertical(mask: np.ndarray) -> np.ndarray:
        """Flip mask vertically.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Vertically flipped mask
        """
        return cv2.flip(mask, 0)
    
    @staticmethod
    def elastic_deformation(mask: np.ndarray, 
                           alpha: float = 34, 
                           sigma: float = 4,
                           random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Apply elastic deformation to mask.
        
        Args:
            mask: Binary mask (H, W)
            alpha: Intensity of deformation
            sigma: Smoothness of deformation (higher = smoother)
            random_state: Random state for reproducibility
            
        Returns:
            Elastically deformed mask
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        shape = mask.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        # Apply deformation
        deformed = map_coordinates(mask, indices, order=1, mode='constant', cval=0)
        deformed = deformed.reshape(shape)
        
        # Ensure binary
        _, deformed = cv2.threshold(deformed.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        return deformed
    
    @staticmethod
    def translate(mask: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """Translate mask by given offsets.
        
        Args:
            mask: Binary mask (H, W)
            tx: Translation in x direction (pixels)
            ty: Translation in y direction (pixels)
            
        Returns:
            Translated mask
        """
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = mask.shape
        translated = cv2.warpAffine(mask, translation_matrix, (w, h),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        return translated
    
    @staticmethod
    def shear(mask: np.ndarray, shear_factor: float, axis: str = 'x') -> np.ndarray:
        """Apply shear transformation to mask.
        
        Args:
            mask: Binary mask (H, W)
            shear_factor: Shear factor
            axis: Shear axis ('x' or 'y')
            
        Returns:
            Sheared mask
        """
        h, w = mask.shape
        
        if axis == 'x':
            # Shear along x-axis
            shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        else:
            # Shear along y-axis
            shear_matrix = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
        
        sheared = cv2.warpAffine(mask, shear_matrix, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        return sheared
    
    @staticmethod
    def apply_random_transform(mask: np.ndarray, 
                              include_elastic: bool = True,
                              random_state: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, str]:
        """Apply a random combination of transformations.
        
        Args:
            mask: Binary mask (H, W)
            include_elastic: Whether to include elastic deformation
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (transformed_mask, transformation_description)
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        result = mask.copy()
        transforms_applied = []
        
        # Rotation (50% chance)
        if random_state.rand() < 0.5:
            angle = random_state.uniform(-180, 180)
            result = MaskTransformer.rotate(result, angle)
            transforms_applied.append(f"rot_{angle:.1f}")
        
        # Scaling (40% chance)
        if random_state.rand() < 0.4:
            scale_factor = random_state.uniform(0.7, 1.3)
            result = MaskTransformer.scale(result, scale_factor)
            transforms_applied.append(f"scale_{scale_factor:.2f}")
        
        # Horizontal flip (30% chance)
        if random_state.rand() < 0.3:
            result = MaskTransformer.flip_horizontal(result)
            transforms_applied.append("flip_h")
        
        # Vertical flip (30% chance)
        if random_state.rand() < 0.3:
            result = MaskTransformer.flip_vertical(result)
            transforms_applied.append("flip_v")
        
        # Translation (20% chance)
        if random_state.rand() < 0.2:
            h, w = result.shape
            tx = random_state.randint(-w//8, w//8)
            ty = random_state.randint(-h//8, h//8)
            result = MaskTransformer.translate(result, tx, ty)
            transforms_applied.append(f"trans_{tx}_{ty}")
        
        # Shear (15% chance)
        if random_state.rand() < 0.15:
            shear_factor = random_state.uniform(-0.2, 0.2)
            axis = random_state.choice(['x', 'y'])
            result = MaskTransformer.shear(result, shear_factor, axis)
            transforms_applied.append(f"shear_{axis}_{shear_factor:.2f}")
        
        # Elastic deformation (30% chance)
        if include_elastic and random_state.rand() < 0.3:
            alpha = random_state.uniform(20, 50)
            sigma = random_state.uniform(3, 6)
            result = MaskTransformer.elastic_deformation(result, alpha, sigma, random_state)
            transforms_applied.append(f"elastic_{alpha:.1f}_{sigma:.1f}")
        
        # Ensure binary output
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        transform_desc = "_".join(transforms_applied) if transforms_applied else "identity"
        return result, transform_desc
