================================================================================
GENERATING TRANSFORMED MASKS
================================================================================
Found 1617 masks in D:\dataset\tgs-salt\train\masks1090

Generating 100 transformed masks...
Output directory: output\generated_masks
Generating masks: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 1022.83it/s]

✓ Generated 100 masks
✓ Saved to: output\generated_masks
✓ Log saved to: output\generated_masks\transformations_log.txt

================================================================================
✓ Generation complete! 100 masks created.
================================================================================

(venv) (base) PS D:\0Code\_phdSeismic\GeomTransf> python src/main.py evaluate --generated output\generated_masks --groundtruth D:\dataset\tgs-salt\train\masks1090 --output results

================================================================================
EVALUATING MASKS
================================================================================

================================================================================
Evaluating mask set: Generated Masks
================================================================================
Found 100 masks in output\generated_masks
Loaded 100 masks

Calculating pairwise IoU for 100 masks...
Computing IoU pairs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4950/4950 [00:00<00:00, 27234.38it/s] 

Calculating shape descriptors for 100 masks...
Computing shape descriptors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 2807.26it/s] 

Results for Generated Masks:
  Masks: 100
  Diversity Score: 0.7796
  Mean IoU: 0.2204
  Coverage: 0.3747

  Shape Statistics:
    Area: mean=3647.20, std=2133.43
    Perimeter: mean=285.65, std=65.55
    Compactness: mean=0.51, std=0.14
    Eccentricity: mean=0.85, std=0.16
    Solidity: mean=0.87, std=0.11
    Extent: mean=0.61, std=0.16

================================================================================
COMPARING WITH GROUND TRUTH
================================================================================

================================================================================
Evaluating mask set: Ground Truth
================================================================================
Found 1617 masks in D:\dataset\tgs-salt\train\masks1090
Loaded 1617 masks

Calculating pairwise IoU for 1617 masks...
Computing IoU pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1306536/1306536 [00:49<00:00, 26459.51it/s]

Calculating shape descriptors for 1617 masks...
Computing shape descriptors: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1617/1617 [00:00<00:00, 25537.56it/s] 

Results for Ground Truth:
  Masks: 1617
  Diversity Score: 0.7134
  Mean IoU: 0.2866
  Coverage: 0.4556

  Shape Statistics:
    Area: mean=4454.88, std=2333.14
    Perimeter: mean=303.37, std=68.79
    Compactness: mean=0.56, std=0.15
    Eccentricity: mean=0.83, std=0.25
    Solidity: mean=0.91, std=0.10
    Extent: mean=0.70, std=0.18
