"""create_subset_by_salt_coverage.py — Cria dataset filtrado preservando test set original.

Estratégia
----------
1. Calcula a cobertura de sal (%) de todas as máscaras.
2. Separa um test set estratificado (SEM filtro) — representa a distribuição real.
3. Aplica o filtro 10–90% APENAS no pool de treino restante.
4. Copia os arquivos para:
     <out_dir>/test/images  + masks/   ← sem filtro, distribuição original
     <out_dir>/train_filtered/images + masks/ ← filtrado 10–90%
5. Salva CSVs de estatísticas.

Uso
---
# Padrão: dataset em D:\\dataset\\tgs-salt\\train, saída em dataset/subset_split
python create_subset_by_salt_coverage.py

# Personalizado
python create_subset_by_salt_coverage.py \\
    --tgs_dir D:\\dataset\\tgs-salt\\train \\
    --out_dir D:\\dataset\\tgs-salt\\subset_split \\
    --test_split 0.20 --min_pct 10 --max_pct 90 --seed 42

# Apenas analisar (sem copiar arquivos)
python create_subset_by_salt_coverage.py --csv_only
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cria subset TGS: test set original + train filtrado 10–90%."
    )
    p.add_argument(
        "--tgs_dir",
        default=r"D:\dataset\tgs-salt\train",
        help="Diretório raiz do TGS (contém subpastas images/ e masks/).",
    )
    p.add_argument(
        "--out_dir",
        default=r"D:\dataset\tgs-salt\subset_split",
        help="Diretório de saída para o subconjunto.",
    )
    p.add_argument(
        "--test_split",
        type=float,
        default=0.20,
        help="Fração do dataset total reservada para o test set (padrão: 0.20).",
    )
    p.add_argument(
        "--min_pct",
        type=float,
        default=10.0,
        help="Cobertura mínima de sal em %% para o train pool (padrão: 10).",
    )
    p.add_argument(
        "--max_pct",
        type=float,
        default=90.0,
        help="Cobertura máxima de sal em %% para o train pool (padrão: 90).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para o split (padrão: 42).",
    )
    p.add_argument(
        "--csv_only",
        action="store_true",
        help="Apenas analisa e salva CSVs, sem copiar arquivos.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------

def salt_coverage_pct(mask_path: str) -> float:
    """Retorna a porcentagem de pixels de sal (brancos > 127) na máscara."""
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Máscara não encontrada: {mask_path}")
    salt_pixels = int(np.count_nonzero(mask > 127))
    return 100.0 * salt_pixels / mask.size


def build_stats(img_dir: Path, mask_dir: Path) -> pd.DataFrame:
    """Calcula cobertura de sal para todas as amostras."""
    mask_paths = sorted(list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")))
    if not mask_paths:
        print(f"[ERRO] Nenhuma máscara encontrada em: {mask_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for mp in tqdm(mask_paths, desc="Calculando cobertura de sal"):
        stem = mp.stem
        img_path = img_dir / (stem + ".png")
        if not img_path.exists():
            img_path = img_dir / (stem + ".jpg")
        if not img_path.exists():
            print(f"  [AVISO] Imagem não encontrada para: {mp.name} — pulando.")
            continue
        try:
            coverage = salt_coverage_pct(str(mp))
        except Exception as e:
            print(f"  [AVISO] Erro ao ler {mp.name}: {e} — pulando.")
            continue
        records.append({
            "id":        stem,
            "img_path":  str(img_path),
            "mask_path": str(mp),
            "salt_pct":  round(coverage, 4),
        })

    return pd.DataFrame(records)


def stratify_label(salt_pct: float) -> int:
    """Rótulo de estratificação: 0 = sem sal, 1 = pouco, 2 = médio, 3 = muito."""
    if salt_pct == 0:
        return 0
    elif salt_pct < 25:
        return 1
    elif salt_pct < 75:
        return 2
    else:
        return 3


def print_summary(df: pd.DataFrame, test_df: pd.DataFrame,
                  train_filtered: pd.DataFrame, min_pct: float, max_pct: float):
    total = len(df)
    train_pool = len(df) - len(test_df)
    excluded = train_pool - len(train_filtered)

    print("\n" + "=" * 65)
    print("  SUMÁRIO DO SPLIT + FILTRAGEM")
    print("=" * 65)
    print(f"  Dataset total            : {total:>6} amostras")
    print(f"  Test set (sem filtro)    : {len(test_df):>6}  ({100*len(test_df)/total:.1f}%)")
    print(f"  Train pool               : {train_pool:>6}  ({100*train_pool/total:.1f}%)")
    print(f"  Filtro aplicado          : {min_pct:.0f}% – {max_pct:.0f}%")
    print(f"  Train filtrado (usável)  : {len(train_filtered):>6}  ({100*len(train_filtered)/total:.1f}% do total)")
    print(f"  Excluídos do train pool  : {excluded:>6}  ({100*excluded/train_pool:.1f}% do pool)")
    print("-" * 65)

    for label, subset, name in [
        ("Test set (original)", test_df, "test"),
        ("Train filtrado", train_filtered, "train"),
    ]:
        print(f"\n  {label} — distribuição por cobertura:")
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
        counts, edges = np.histogram(subset["salt_pct"], bins=bins)
        max_cnt = max(counts) if max(counts) > 0 else 1
        for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
            bar = "█" * int(30 * cnt / max_cnt)
            hi_str = "100" if hi > 100 else f"{hi:.0f}"
            print(f"    {lo:>3.0f}–{hi_str:>3}%  {cnt:>5}  {bar}")

    print("=" * 65 + "\n")


def copy_subset(df: pd.DataFrame, out_img_dir: Path, out_mask_dir: Path, label: str):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copiando {label}"):
        shutil.copy2(row["img_path"],  out_img_dir  / Path(row["img_path"]).name)
        shutil.copy2(row["mask_path"], out_mask_dir / Path(row["mask_path"]).name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    tgs_dir  = Path(args.tgs_dir)
    img_dir  = tgs_dir / "images"
    mask_dir = tgs_dir / "masks"
    out_dir  = Path(args.out_dir)

    print(f"\n[INFO] TGS dir     : {tgs_dir}")
    print(f"[INFO] Saída       : {out_dir}")
    print(f"[INFO] Test split  : {args.test_split:.0%}")
    print(f"[INFO] Filtro train: {args.min_pct:.0f}% – {args.max_pct:.0f}%")
    print(f"[INFO] Seed        : {args.seed}")
    print(f"[INFO] Modo        : {'apenas CSV (--csv_only)' if args.csv_only else 'copiar arquivos'}\n")

    # ------------------------------------------------------------------
    # 1. Calcular cobertura de todas as amostras
    # ------------------------------------------------------------------
    df = build_stats(img_dir, mask_dir)
    print(f"\n[INFO] Total de amostras válidas: {len(df)}")

    # ------------------------------------------------------------------
    # 2. Split estratificado: test set SEM filtro
    # ------------------------------------------------------------------
    df["strat_label"] = df["salt_pct"].apply(stratify_label)

    train_pool_df, test_df = train_test_split(
        df,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=df["strat_label"],
    )
    train_pool_df = train_pool_df.reset_index(drop=True)
    test_df       = test_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Filtrar train pool: 10–90%
    # ------------------------------------------------------------------
    train_filtered = train_pool_df[
        (train_pool_df["salt_pct"] >= args.min_pct) &
        (train_pool_df["salt_pct"] <= args.max_pct)
    ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Sumário
    # ------------------------------------------------------------------
    print_summary(df, test_df, train_filtered, args.min_pct, args.max_pct)

    # ------------------------------------------------------------------
    # 5. Salvar CSVs
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # Marcar split em CSV consolidado
    df["split"] = "train_excluded"
    df.loc[df["id"].isin(test_df["id"]), "split"] = "test"
    df.loc[df["id"].isin(train_filtered["id"]), "split"] = "train_filtered"

    csv_all      = out_dir / "split_stats.csv"
    csv_test     = out_dir / "test_set.csv"
    csv_train    = out_dir / "train_filtered.csv"

    df.to_csv(csv_all,   index=False)
    test_df.to_csv(csv_test,  index=False)
    train_filtered.to_csv(csv_train, index=False)

    print(f"[INFO] CSVs salvos em: {out_dir}")
    print(f"       split_stats.csv    — todas as {len(df)} amostras com rótulo de split")
    print(f"       test_set.csv       — {len(test_df)} amostras do test set")
    print(f"       train_filtered.csv — {len(train_filtered)} amostras de treino filtradas")

    # ------------------------------------------------------------------
    # 6. Copiar arquivos
    # ------------------------------------------------------------------
    if not args.csv_only:
        print()
        copy_subset(
            test_df,
            out_dir / "test"  / "images",
            out_dir / "test"  / "masks",
            label="test set",
        )
        copy_subset(
            train_filtered,
            out_dir / "train_filtered" / "images",
            out_dir / "train_filtered" / "masks",
            label="train filtrado",
        )
        print(f"\n[OK] Estrutura criada em: {out_dir}")
        print(f"     test/            → {len(test_df)} pares (sem filtro)")
        print(f"     train_filtered/  → {len(train_filtered)} pares (filtro {args.min_pct:.0f}–{args.max_pct:.0f}%)")

    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
