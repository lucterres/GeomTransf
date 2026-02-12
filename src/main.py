"""Main module for GeomTransf application."""

import argparse
import sys
from pathlib import Path

# Fix imports for running as script
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mask_generator import MaskGenerator
from src.evaluator import MaskEvaluator
from src.mask_loader import MaskLoader


def generate_masks(args):
    """Generate transformed masks."""
    print("\n" + "="*80)
    print("GENERATING TRANSFORMED MASKS")
    print("="*80)
    
    # Create generator
    generator = MaskGenerator(
        input_dir=args.input,
        output_dir=args.output,
        seed=args.seed
    )
    
    # Generate masks
    generated_files = generator.generate(
        count=args.count,
        include_elastic=not args.no_elastic
    )
    
    # Validate generated masks
    if args.validate:
        generator.validate_generated_masks()
    
    print(f"\n{'='*80}")
    print(f"✓ Generation complete! {len(generated_files)} masks created.")
    print(f"{'='*80}\n")


def evaluate_masks(args):
    """Evaluate a set of masks."""
    print("\n" + "="*80)
    print("EVALUATING MASKS")
    print("="*80)
    
    evaluator = MaskEvaluator()
    
    # Evaluate generated masks
    results = evaluator.evaluate_mask_set(
        masks_dir=args.generated,
        set_name="Generated Masks",
        max_masks=args.max_masks
    )
    
    # Compare with ground truth if provided
    if args.groundtruth:
        print("\n" + "="*80)
        print("COMPARING WITH GROUND TRUTH")
        print("="*80)
        
        gt_results = evaluator.evaluate_mask_set(
            masks_dir=args.groundtruth,
            set_name="Ground Truth",
            max_masks=args.max_masks
        )
    
    # Save results
    if args.output:
        output_file = Path(args.output) / "evaluation_results.json"
        evaluator.save_results(str(output_file))


def compare_sets(args):
    """Compare two mask sets."""
    print("\n" + "="*80)
    print("COMPARING MASK SETS - ABLATION STUDY")
    print("="*80)
    
    evaluator = MaskEvaluator()
    
    comparison = evaluator.compare_sets(
        set1_dir=args.geometric,
        set2_dir=args.vae,
        groundtruth_dir=args.groundtruth,
        set1_name="Geometric Transforms",
        set2_name="VAE Generated"
    )
    
    # Save comparison results
    if args.output:
        output_file = Path(args.output) / "comparison_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✓ Comparison results saved to: {output_file}")


def validate_dataset(args):
    """Validate mask dataset."""
    print("\n" + "="*80)
    print("VALIDATING DATASET")
    print("="*80)
    
    loader = MaskLoader(args.input)
    
    print(f"\nDataset location: {args.input}")
    print(f"Number of masks: {len(loader.mask_files)}")
    print(f"Mask shape: {loader.get_mask_shape()}")
    
    if loader.validate_masks():
        print("\n✓ Dataset is valid!")
    else:
        print("\n✗ Dataset validation failed!")
        sys.exit(1)


def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser(
        description="GeomTransf - Geometric Transformations for Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 transformed masks
  python src/main.py generate --input D:\\dataset\\tgs-salt\\train\\masks --output output\\generated_masks --count 100

  # Evaluate generated masks
  python src/main.py evaluate --generated output\\generated_masks --groundtruth D:\\dataset\\tgs-salt\\train\\masks --output results

  # Compare geometric vs VAE
  python src/main.py compare --geometric output\\generated_masks --vae path\\to\\vae\\masks --groundtruth D:\\dataset\\tgs-salt\\train\\masks --output results

  # Validate dataset
  python src/main.py validate --input D:\\dataset\\tgs-salt\\train\\masks
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate transformed masks')
    gen_parser.add_argument('--input', required=True, 
                           help='Input directory with ground-truth masks')
    gen_parser.add_argument('--output', required=True,
                           help='Output directory for generated masks')
    gen_parser.add_argument('--count', type=int, default=100,
                           help='Number of masks to generate (default: 100)')
    gen_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    gen_parser.add_argument('--no-elastic', action='store_true',
                           help='Disable elastic deformation')
    gen_parser.add_argument('--validate', action='store_true',
                           help='Validate generated masks after creation')
    gen_parser.set_defaults(func=generate_masks)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate mask set')
    eval_parser.add_argument('--generated', required=True,
                            help='Directory with generated masks')
    eval_parser.add_argument('--groundtruth',
                            help='Directory with ground-truth masks for comparison')
    eval_parser.add_argument('--output', default='results',
                            help='Output directory for results (default: results)')
    eval_parser.add_argument('--max-masks', type=int,
                            help='Maximum number of masks to evaluate')
    eval_parser.set_defaults(func=evaluate_masks)
    
    # Compare command
    comp_parser = subparsers.add_parser('compare', help='Compare two mask sets')
    comp_parser.add_argument('--geometric', required=True,
                            help='Directory with geometric transformed masks')
    comp_parser.add_argument('--vae', required=True,
                            help='Directory with VAE generated masks')
    comp_parser.add_argument('--groundtruth', required=True,
                            help='Directory with ground-truth masks')
    comp_parser.add_argument('--output', default='results',
                            help='Output directory for results (default: results)')
    comp_parser.set_defaults(func=compare_sets)
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate mask dataset')
    val_parser.add_argument('--input', required=True,
                           help='Directory with masks to validate')
    val_parser.set_defaults(func=validate_dataset)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
