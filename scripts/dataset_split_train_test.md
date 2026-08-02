"""create_subset_by_salt_coverage.py — Filtra o dataset TGS por cobertura de sal.

Seleciona apenas amostras cuja máscara tem entre MIN_PCT% e MAX_PCT% de pixels
de sal, copiando as imagens e máscaras correspondentes para um novo diretório.

Uso
---
# Filtro padrão: 10–90 %
python create_subset_by_salt_coverage.py \
    --tgs_dir /var/tmp/cym7/datasets/tgs-salt/train \
    --out_dir /var/tmp/cym7/datasets/tgs-salt/subset_10_90

# Filtro personalizado
python create_subset_by_salt_coverage.py \
    --tgs_dir /var/tmp/cym7/datasets/tgs-salt/train \
    --out_dir dataset/subset_10_90 \
    --min_pct 10 --max_pct 90
"""
