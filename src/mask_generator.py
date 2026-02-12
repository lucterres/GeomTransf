"""Generate transformed masks for ablation study."""

import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from tqdm import tqdm

from src.mask_loader import MaskLoader
from src.transformations import MaskTransformer


class MaskGenerator:
    """Generate masks using geometric transformations."""
    
    def __init__(self, input_dir: str, output_dir: str, seed: Optional[int] = 42):
        """Initialize mask generator.
        
        Args:
            input_dir: Directory with ground-truth masks
            output_dir: Directory to save generated masks
            seed: Random seed for reproducibility
        """
        self.loader = MaskLoader(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        
    def generate(self, count: int = 100, include_elastic: bool = True) -> List[str]:
        """Generate transformed masks.
        
        Args:
            count: Number of masks to generate
            include_elastic: Whether to include elastic deformation
            
        Returns:
            List of generated mask file paths
        """
        print(f"\nGenerating {count} transformed masks...")
        print(f"Output directory: {self.output_dir}")
        
        # Load random source masks
        source_masks = self.loader.load_random_masks(count, seed=self.seed)
        
        generated_files = []
        transform_log = []
        
        for i, (mask_id, mask) in enumerate(tqdm(source_masks, desc="Generating masks")):
            # Apply random transformation
            transformed_mask, transform_desc = MaskTransformer.apply_random_transform(
                mask, 
                include_elastic=include_elastic,
                random_state=self.random_state
            )
            
            # Generate output filename
            output_filename = f"generated_{i:04d}.png"
            output_path = self.output_dir / output_filename
            
            # Save transformed mask
            cv2.imwrite(str(output_path), transformed_mask)
            generated_files.append(str(output_path))
            
            # Log transformation
            transform_log.append({
                'index': i,
                'output_file': output_filename,
                'source_mask': mask_id,
                'transforms': transform_desc
            })
        
        # Save transformation log
        log_path = self.output_dir / "transformations_log.txt"
        self._save_log(log_path, transform_log)
        
        print(f"\n✓ Generated {len(generated_files)} masks")
        print(f"✓ Saved to: {self.output_dir}")
        print(f"✓ Log saved to: {log_path}")
        
        return generated_files
    
    def _save_log(self, log_path: Path, transform_log: List[dict]):
        """Save transformation log to file.
        
        Args:
            log_path: Path to save log file
            transform_log: List of transformation records
        """
        with open(log_path, 'w') as f:
            f.write("Geometric Transformations Log\n")
            f.write("="*80 + "\n")
            f.write(f"Random Seed: {self.seed}\n")
            f.write(f"Total Masks: {len(transform_log)}\n")
            f.write("="*80 + "\n\n")
            
            for record in transform_log:
                f.write(f"ID: {record['index']:04d}\n")
                f.write(f"Output: {record['output_file']}\n")
                f.write(f"Source: {record['source_mask']}\n")
                f.write(f"Transforms: {record['transforms']}\n")
                f.write("-"*80 + "\n")
    
    def generate_specific_transforms(self, mask: np.ndarray, 
                                    output_prefix: str = "test") -> List[str]:
        """Generate masks with specific transformations for testing.
        
        Args:
            mask: Source mask
            output_prefix: Prefix for output filenames
            
        Returns:
            List of generated file paths
        """
        generated_files = []
        
        transforms = [
            ("original", mask),
            ("rotate_45", MaskTransformer.rotate(mask, 45)),
            ("rotate_90", MaskTransformer.rotate(mask, 90)),
            ("scale_0.8", MaskTransformer.scale(mask, 0.8)),
            ("scale_1.2", MaskTransformer.scale(mask, 1.2)),
            ("flip_h", MaskTransformer.flip_horizontal(mask)),
            ("flip_v", MaskTransformer.flip_vertical(mask)),
            ("elastic", MaskTransformer.elastic_deformation(mask, alpha=40, sigma=4)),
        ]
        
        for name, transformed in transforms:
            output_filename = f"{output_prefix}_{name}.png"
            output_path = self.output_dir / output_filename
            cv2.imwrite(str(output_path), transformed)
            generated_files.append(str(output_path))
        
        print(f"Generated {len(generated_files)} test transforms")
        return generated_files
    
    def validate_generated_masks(self) -> bool:
        """Validate generated masks.
        
        Returns:
            True if all masks are valid
        """
        mask_files = sorted(list(self.output_dir.glob("generated_*.png")))
        
        if not mask_files:
            print("No generated masks found")
            return False
        
        print(f"Validating {len(mask_files)} generated masks...")
        
        for filepath in mask_files:
            mask = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load: {filepath.name}")
                return False
            
            unique_values = np.unique(mask)
            if not np.all(np.isin(unique_values, [0, 255])):
                print(f"Non-binary values in: {filepath.name}")
                return False
        
        print("✓ All generated masks are valid")
        return True
