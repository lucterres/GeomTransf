# Quick Start Guide

## Projeto GeomTransf - Ablation Study

Este projeto gera 100 máscaras usando transformações geométricas para comparação com máscaras geradas por VAE.

## Execução Rápida

### 1. Gerar 100 Máscaras Transformadas
```bash
python src/main.py generate --input D:\dataset\tgs-salt\train\masks --output output\generated_masks --count 100
```

### 2. Avaliar Máscaras Geradas
```bash
python src/main.py evaluate --generated output\generated_masks --groundtruth D:\dataset\tgs-salt\train\masks --output results
```

### 3. Comparar com Máscaras VAE (quando disponível)
```bash
python src/main.py compare --geometric output\generated_masksGeomTrans1090 --vae output\generated_masksVae1090 --groundtruth D:/dataset/tgs-salt/train/masks1090 --output results
```

## Métricas Calculadas

### 1. Diversidade (Shape Diversity)
- **IoU Pairwise**: Calcula IoU entre todos os pares de máscaras
- **Diversity Score**: 1 - mean_IoU (maior = mais diverso)

### 2. Realismo Geométrico (Geometric Realism)
Shape descriptors calculados:
- **Area**: Área da região
- **Perimeter**: Perímetro da região
- **Compactness**: Circularidade (4π*area / perimeter²)
- **Eccentricity**: Excentricidade da elipse ajustada
- **Solidity**: Área / área do convex hull
- **Extent**: Área / área do bounding box

### 3. Teste Estatístico
- **Kolmogorov-Smirnov Test**: Compara distribuições de shape descriptors entre máscaras geradas e ground truth

## Transformações Aplicadas

As máscaras são geradas com combinações aleatórias de:
- Rotação (ângulos variados)
- Escala (0.7x a 1.3x)
- Flip horizontal/vertical
- Translação
- Shear
- Deformação elástica

## Arquivos de Saída

- `output/generated_masks/*.png`: 100 máscaras geradas
- `output/generated_masks/transformations_log.txt`: Log das transformações aplicadas
- `results/evaluation_results.json`: Resultados das métricas
- `results/comparison_results.json`: Resultados da comparação (quando usar compare)

## Resultados Atuais

Após a execução de teste:
- **Máscaras geradas**: 100 ✓
- **Diversity Score**: 0.9410 (alto = boa diversidade)
- **Mean IoU**: 0.0590 (baixo = máscaras bem diferentes)
- **Coverage médio**: 21.75%

Comparação com Ground Truth:
- Ground Truth Diversity: 0.9279
- Máscaras geométricas estão ligeiramente **mais diversas** que o GT
