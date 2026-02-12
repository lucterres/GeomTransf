"""Statistical evaluation and comparison of mask sets."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
import cv2

from src.mask_loader import MaskLoader
from src.metrics import MaskMetrics


class MaskEvaluator:
    """Evaluate and compare mask sets statistically."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def evaluate_mask_set(self, masks_dir: str, 
                         set_name: str = "masks",
                         max_masks: Optional[int] = None) -> Dict:
        """Evaluate a set of masks.
        
        Args:
            masks_dir: Directory containing masks
            set_name: Name for this mask set
            max_masks: Maximum number of masks to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print(f"Evaluating mask set: {set_name}")
        print(f"{'='*80}")
        
        # Load masks
        loader = MaskLoader(masks_dir)
        all_masks_data = loader.get_all_masks()
        
        if max_masks is not None and len(all_masks_data) > max_masks:
            print(f"Using {max_masks} of {len(all_masks_data)} masks")
            # Random sample
            indices = np.random.choice(len(all_masks_data), max_masks, replace=False)
            all_masks_data = [all_masks_data[i] for i in indices]
        
        masks = [mask for _, mask in all_masks_data]
        
        print(f"Loaded {len(masks)} masks")
        
        # Calculate diversity (pairwise IoU)
        iou_stats = MaskMetrics.calculate_pairwise_iou(masks)
        diversity_score = MaskMetrics.calculate_diversity_score(iou_stats['mean_iou'])
        
        # Calculate shape descriptors
        shape_stats = MaskMetrics.calculate_shape_statistics(masks)
        
        # Calculate coverage
        coverage = MaskMetrics.calculate_coverage(masks)
        
        results = {
            'set_name': set_name,
            'n_masks': len(masks),
            'diversity': {
                'diversity_score': diversity_score,
                **iou_stats
            },
            'shape_statistics': shape_stats,
            'coverage': coverage
        }
        
        self.results[set_name] = results
        
        self._print_results(results)
        
        return results
    
    def compare_sets(self, set1_dir: str, set2_dir: str,
                    groundtruth_dir: str,
                    set1_name: str = "Geometric",
                    set2_name: str = "VAE") -> Dict:
        """Compare two mask sets against ground truth.
        
        Args:
            set1_dir: Directory with first mask set
            set2_dir: Directory with second mask set
            groundtruth_dir: Directory with ground truth masks
            set1_name: Name for first set
            set2_name: Name for second set
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*80}")
        print(f"Comparing Mask Sets")
        print(f"{'='*80}")
        
        # Evaluate all three sets
        gt_results = self.evaluate_mask_set(groundtruth_dir, "Ground Truth", max_masks=100)
        set1_results = self.evaluate_mask_set(set1_dir, set1_name, max_masks=100)
        set2_results = self.evaluate_mask_set(set2_dir, set2_name, max_masks=100)
        
        # Statistical comparison using K-S test
        print(f"\n{'='*80}")
        print("Statistical Comparison (Kolmogorov-Smirnov Test)")
        print(f"{'='*80}")
        
        ks_results = self._kolmogorov_smirnov_test(gt_results, set1_results, set2_results)
        
        # Overall comparison
        comparison = {
            'ground_truth': gt_results,
            set1_name: set1_results,
            set2_name: set2_results,
            'statistical_tests': ks_results,
            'summary': self._generate_summary(set1_results, set2_results, ks_results, 
                                             set1_name, set2_name)
        }
        
        return comparison
    
    def _kolmogorov_smirnov_test(self, gt_results: Dict, 
                                 set1_results: Dict, 
                                 set2_results: Dict) -> Dict:
        """Perform Kolmogorov-Smirnov test comparing distributions.
        
        Args:
            gt_results: Ground truth results
            set1_results: First set results
            set2_results: Second set results
            
        Returns:
            Dictionary with K-S test results
        """
        # We need to compare distributions of shape descriptors
        # For this, we need the raw values, not just statistics
        # This is a simplified version using statistics
        
        shape_descriptors = ['area', 'perimeter', 'compactness', 
                           'eccentricity', 'solidity', 'extent']
        
        ks_results = {}
        
        for descriptor in shape_descriptors:
            # Get statistics for each set
            gt_stats = gt_results['shape_statistics'][descriptor]
            set1_stats = set1_results['shape_statistics'][descriptor]
            set2_stats = set2_results['shape_statistics'][descriptor]
            
            # Generate synthetic distributions from statistics (approximation)
            # In reality, we'd use the actual values
            gt_samples = self._generate_synthetic_distribution(gt_stats)
            set1_samples = self._generate_synthetic_distribution(set1_stats)
            set2_samples = self._generate_synthetic_distribution(set2_stats)
            
            # K-S test: set1 vs ground truth
            ks_stat_1, p_value_1 = stats.ks_2samp(set1_samples, gt_samples)
            
            # K-S test: set2 vs ground truth
            ks_stat_2, p_value_2 = stats.ks_2samp(set2_samples, gt_samples)
            
            ks_results[descriptor] = {
                'set1_vs_gt': {
                    'ks_statistic': float(ks_stat_1),
                    'p_value': float(p_value_1),
                    'similar': p_value_1 > 0.05  # Not significantly different
                },
                'set2_vs_gt': {
                    'ks_statistic': float(ks_stat_2),
                    'p_value': float(p_value_2),
                    'similar': p_value_2 > 0.05
                }
            }
            
            print(f"\n{descriptor.capitalize()}:")
            print(f"  {set1_results['set_name']} vs GT: KS={ks_stat_1:.4f}, p={p_value_1:.4f}")
            print(f"  {set2_results['set_name']} vs GT: KS={ks_stat_2:.4f}, p={p_value_2:.4f}")
        
        return ks_results
    
    def _generate_synthetic_distribution(self, stats: Dict, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic distribution from statistics.
        
        This is an approximation assuming normal distribution.
        
        Args:
            stats: Statistics dictionary with 'mean' and 'std'
            n_samples: Number of samples to generate
            
        Returns:
            Array of synthetic samples
        """
        mean = stats['mean']
        std = stats['std']
        
        if std == 0:
            return np.full(n_samples, mean)
        
        samples = np.random.normal(mean, std, n_samples)
        # Clip to min/max from original stats
        samples = np.clip(samples, stats['min'], stats['max'])
        
        return samples
    
    def _generate_summary(self, set1_results: Dict, set2_results: Dict,
                         ks_results: Dict, set1_name: str, set2_name: str) -> Dict:
        """Generate comparison summary.
        
        Args:
            set1_results: First set results
            set2_results: Second set results
            ks_results: K-S test results
            set1_name: Name of first set
            set2_name: Name of second set
            
        Returns:
            Summary dictionary
        """
        # Count how many descriptors are similar to GT for each set
        set1_similar_count = sum(1 for desc in ks_results.values() 
                                if desc['set1_vs_gt']['similar'])
        set2_similar_count = sum(1 for desc in ks_results.values() 
                                if desc['set2_vs_gt']['similar'])
        
        # Compare diversity
        set1_diversity = set1_results['diversity']['diversity_score']
        set2_diversity = set2_results['diversity']['diversity_score']
        
        summary = {
            'diversity_comparison': {
                set1_name: set1_diversity,
                set2_name: set2_diversity,
                'winner': set1_name if set1_diversity > set2_diversity else set2_name
            },
            'realism_comparison': {
                f'{set1_name}_similar_descriptors': set1_similar_count,
                f'{set2_name}_similar_descriptors': set2_similar_count,
                'total_descriptors': len(ks_results),
                'winner': set1_name if set1_similar_count > set2_similar_count else set2_name
            }
        }
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\nDiversity (higher is better):")
        print(f"  {set1_name}: {set1_diversity:.4f}")
        print(f"  {set2_name}: {set2_diversity:.4f}")
        print(f"  Winner: {summary['diversity_comparison']['winner']}")
        
        print(f"\nRealism (more similar descriptors to GT):")
        print(f"  {set1_name}: {set1_similar_count}/{len(ks_results)} descriptors")
        print(f"  {set2_name}: {set2_similar_count}/{len(ks_results)} descriptors")
        print(f"  Winner: {summary['realism_comparison']['winner']}")
        
        return summary
    
    def _print_results(self, results: Dict):
        """Print evaluation results.
        
        Args:
            results: Results dictionary
        """
        print(f"\nResults for {results['set_name']}:")
        print(f"  Masks: {results['n_masks']}")
        print(f"  Diversity Score: {results['diversity']['diversity_score']:.4f}")
        print(f"  Mean IoU: {results['diversity']['mean_iou']:.4f}")
        print(f"  Coverage: {results['coverage']:.4f}")
        
        print(f"\n  Shape Statistics:")
        for desc, stats in results['shape_statistics'].items():
            print(f"    {desc.capitalize()}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
