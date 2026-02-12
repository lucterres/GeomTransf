# GeomTransf - Geometric Transformations for Ablation Study

Generate 100 masks using geometric transformations for comparison with VAE-generated masks.

## Setup

1. Activate virtual environment:
```bash
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate 100 transformed masks
```bash
python src/main.py generate --input D:\dataset\tgs-salt\train\masks --output output\generated_masks --count 100
```

### Calculate metrics
```bash
python src/main.py metrics --generated output\generated_masks --groundtruth D:\dataset\tgs-salt\train\masks --output results
```

### Compare with VAE results
```bash
python src/main.py compare --geometric output\generated_masks --vae path\to\vae\masks --groundtruth D:\dataset\tgs-salt\train\masks
```

## Transformations Applied

- Rotation (random angles)
- Scaling (various factors)
- Horizontal/Vertical flipping
- Elastic deformation

## Metrics Evaluated

- Shape diversity (pairwise IoU)
- Shape descriptors (area, perimeter, compactness, eccentricity)
- Statistical comparison (Kolmogorov-Smirnov test)
