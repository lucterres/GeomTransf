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

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cria subconjunto do TGS filtrando por cobertura de sal."
    )
    p.add_argument(
        "--tgs_dir",
        default=os.environ.get("TGS_PATH", "/var/tmp/cym7/datasets/tgs-salt/train"),
        help="Diretório raiz do TGS (contém subpastas images/ e masks/).",
    )
    p.add_argument(
        "--out_dir",
        default="dataset/subset_10_90",
        help="Diretório de saída para o subconjunto filtrado.",
    )
    p.add_argument(
        "--min_pct",
        type=float,
        default=10.0,
        help="Cobertura mínima de sal em %% (padrão: 10).",
    )
    p.add_argument(
        "--max_pct",
        type=float,
        default=90.0,
        help="Cobertura máxima de sal em %% (padrão: 90).",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copiar arquivos (padrão). Use --no-copy para apenas gerar o CSV.",
    )
    p.add_argument(
        "--no-copy",
        dest="copy",
        action="store_false",
        help="Apenas gera o CSV de estatísticas, sem copiar arquivos.",
    )
    p.add_argument(
        "--csv_only",
        action="store_true",
        help="Atalho para --no-copy: apenas analisa e salva o CSV.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------

def salt_coverage_pct(mask_path: str) -> float:
    """Retorna a porcentagem de pixels de sal (brancos) na máscara."""
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Máscara não encontrada: {mask_path}")
    total = mask.size
    salt_pixels = np.count_nonzero(mask > 127)
    return 100.0 * salt_pixels / total


def build_stats(img_dir: Path, mask_dir: Path) -> pd.DataFrame:
    """Calcula cobertura de sal para todas as amostras."""
    mask_paths = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
    if not mask_paths:
        print(f"[ERRO] Nenhuma máscara encontrada em: {mask_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for mp in tqdm(mask_paths, desc="Calculando cobertura"):
        stem = mp.stem
        # Tenta encontrar a imagem correspondente (png ou jpg)
        img_path = img_dir / (stem + ".png")
        if not img_path.exists():
            img_path = img_dir / (stem + ".jpg")
        if not img_path.exists():
            print(f"[AVISO] Imagem não encontrada para máscara: {mp.name}; pulando.")
            continue
        coverage = salt_coverage_pct(str(mp))
        records.append(
            {"id": stem, "img_path": str(img_path), "mask_path": str(mp), "salt_pct": coverage}
        )

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame, filtered: pd.DataFrame, min_pct: float, max_pct: float):
    """Imprime sumário no terminal."""
    print("\n" + "=" * 60)
    print("  SUMÁRIO DA FILTRAGEM POR COBERTURA DE SAL")
    print("=" * 60)
    print(f"  Amostras totais          : {len(df):>6}")
    print(f"  Cobertura 0%  (sem sal)  : {(df['salt_pct'] == 0).sum():>6}")
    print(f"  Cobertura 100% (sal total): {(df['salt_pct'] == 100).sum():>6}")
    print(f"  Filtro aplicado          : {min_pct:.1f}% – {max_pct:.1f}%")
    print(f"  Amostras selecionadas    : {len(filtered):>6}  "
          f"({100*len(filtered)/len(df):.1f}% do total)")
    print(f"  Amostras excluídas       : {len(df)-len(filtered):>6}")
    print("-" * 60)
    print("  Distribuição das amostras selecionadas:")
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    counts, edges = np.histogram(filtered["salt_pct"], bins=bins)
    for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
        bar = "█" * (cnt // max(1, len(filtered) // 40))
        print(f"    {lo:>3.0f}–{hi:>3.0f}%  {cnt:>5}  {bar}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.csv_only:
        args.copy = False

    tgs_dir  = Path(args.tgs_dir)
    img_dir  = tgs_dir / "images"
    mask_dir = tgs_dir / "masks"
    out_dir  = Path(args.out_dir)

    print(f"\n[INFO] TGS dir  : {tgs_dir}")
    print(f"[INFO] Saída    : {out_dir}")
    print(f"[INFO] Filtro   : {args.min_pct:.1f}% ≤ cobertura ≤ {args.max_pct:.1f}%")
    print(f"[INFO] Copiar   : {'sim' if args.copy else 'não (apenas CSV)'}\n")

    # 1. Calcular cobertura de todas as amostras
    df = build_stats(img_dir, mask_dir)

    # 2. Filtrar
    filtered = df[
        (df["salt_pct"] >= args.min_pct) & (df["salt_pct"] <= args.max_pct)
    ].reset_index(drop=True)

    # 3. Imprimir sumário
    print_summary(df, filtered, args.min_pct, args.max_pct)

    # 4. Salvar CSV de estatísticas (sempre)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_all      = out_dir / "all_coverage.csv"
    csv_filtered = out_dir / "filtered_coverage.csv"
    df.to_csv(csv_all, index=False)
    filtered.to_csv(csv_filtered, index=False)
    print(f"[INFO] CSV completo   salvo em: {csv_all}")
    print(f"[INFO] CSV filtrado   salvo em: {csv_filtered}")

    # 5. Copiar arquivos (opcional)
    if args.copy and len(filtered) > 0:
        out_img_dir  = out_dir / "images"
        out_mask_dir = out_dir / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Copiando {len(filtered)} pares para {out_dir} ...")
        for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Copiando"):
            shutil.copy2(row["img_path"],  out_img_dir  / Path(row["img_path"]).name)
            shutil.copy2(row["mask_path"], out_mask_dir / Path(row["mask_path"]).name)

        print(f"\n[OK] Subconjunto criado:")
        print(f"     {out_img_dir}  ({len(list(out_img_dir.iterdir()))} imagens)")
        print(f"     {out_mask_dir} ({len(list(out_mask_dir.iterdir()))} máscaras)")
    elif args.copy and len(filtered) == 0:
        print("[AVISO] Nenhuma amostra passou pelo filtro. Nada foi copiado.")

    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
