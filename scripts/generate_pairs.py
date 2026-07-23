"""
generate_pairs.py
-----------------
Gera 1600 pares (imagem + máscara) aplicando as MESMAS transformações geométricas
a cada par original de D:\\dataset\\tgs-salt\\train\\images e ...\\masks.

Saída:
    output/pairs1600/images/   → imagens transformadas
    output/pairs1600/masks/    → máscaras transformadas (mesma transformação)
    output/pairs1600/pairs_log.csv → registro de cada par gerado

Uso rápido:
    python scripts/generate_pairs.py

Parâmetros opcionais (linha de comando):
    --images    caminho dos originais de imagem   (default: D:\\dataset\\tgs-salt\\train\\images)
    --masks     caminho dos originais de máscara  (default: D:\\dataset\\tgs-salt\\train\\masks)
    --output    diretório de saída                (default: output/pairs1600)
    --count     número de pares a gerar           (default: 1600)
    --seed      semente aleatória                 (default: 42)
    --no-elastic  desativa deformação elástica
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Transformações geométricas (auto-contidas, sem dependência de src/)
# ---------------------------------------------------------------------------

def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    flags = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def scale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, factor)
    flags = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def flip_h(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def flip_v(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def translate(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    flags = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shear(img: np.ndarray, factor: float, axis: str = 'x') -> np.ndarray:
    h, w = img.shape[:2]
    if axis == 'x':
        M = np.float32([[1, factor, 0], [0, 1, 0]])
    else:
        M = np.float32([[1, 0, 0], [factor, 1, 0]])
    flags = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def elastic_deformation(img: np.ndarray,
                        alpha: float,
                        sigma: float,
                        rs: np.random.RandomState) -> np.ndarray:
    """Deformação elástica. Funciona com imagem grayscale (H,W) ou RGB (H,W,3)."""
    shape_hw = img.shape[:2]

    dx = gaussian_filter((rs.rand(*shape_hw) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rs.rand(*shape_hw) * 2 - 1), sigma) * alpha

    xg, yg = np.meshgrid(np.arange(shape_hw[1]), np.arange(shape_hw[0]))
    indices = (yg + dy).reshape(-1), (xg + dx).reshape(-1)

    if img.ndim == 2:
        deformed = map_coordinates(img, indices, order=1, mode='constant', cval=0)
        return deformed.reshape(shape_hw).astype(img.dtype)
    else:
        channels = []
        for c in range(img.shape[2]):
            ch = map_coordinates(img[:, :, c], indices, order=1, mode='constant', cval=0)
            channels.append(ch.reshape(shape_hw))
        return np.stack(channels, axis=2).astype(img.dtype)


# ---------------------------------------------------------------------------
# Aplicação de transformação aleatória sobre um PAR (img, mask) com o
# mesmo random_state — garantia de identidade exata da transformação.
# ---------------------------------------------------------------------------

def apply_random_transform_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rs: np.random.RandomState,
    include_elastic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Aplica a mesma sequência de transformações geométricas à imagem e à máscara.

    Args:
        image:  Imagem original (H, W) ou (H, W, 3)
        mask:   Máscara binária  (H, W)
        rs:     RandomState compartilhado — MESMO estado para imagem e máscara
        include_elastic: Incluir deformação elástica

    Returns:
        (imagem_transformada, mascara_transformada, descricao_das_transformacoes)
    """

    # ---- Sorteia parâmetros ------------------------------------------------
    # Cada rand()/randint() avança o estado; fazemos TODOS os sorteios ANTES
    # de aplicar qualquer transformação, para que imagem e máscara recebam
    # exatamente os mesmos parâmetros.

    do_rotate   = rs.rand() < 0.6
    angle       = rs.uniform(-180, 180)

    do_scale    = rs.rand() < 0.4
    scale_f     = rs.uniform(0.75, 1.25)

    do_flip_h   = rs.rand() < 0.35
    do_flip_v   = rs.rand() < 0.25

    do_translate = rs.rand() < 0.3
    h, w = mask.shape[:2]
    tx = int(rs.randint(-w // 8, w // 8 + 1))
    ty = int(rs.randint(-h // 8, h // 8 + 1))

    do_shear    = rs.rand() < 0.2
    shear_f     = rs.uniform(-0.2, 0.2)
    shear_axis  = 'x' if rs.rand() < 0.5 else 'y'

    do_elastic  = include_elastic and (rs.rand() < 0.35)
    alpha_e     = rs.uniform(20, 50)
    sigma_e     = rs.uniform(3, 6)

    # ---- Aplica à imagem e à máscara (mesmos parâmetros) -------------------
    img_out  = image.copy()
    mask_out = mask.copy()
    tags: List[str] = []

    if do_rotate:
        img_out  = rotate(img_out,  angle)
        mask_out = rotate(mask_out, angle)
        tags.append(f"rot_{angle:.1f}")

    if do_scale:
        img_out  = scale(img_out,  scale_f)
        mask_out = scale(mask_out, scale_f)
        tags.append(f"scale_{scale_f:.2f}")

    if do_flip_h:
        img_out  = flip_h(img_out)
        mask_out = flip_h(mask_out)
        tags.append("flip_h")

    if do_flip_v:
        img_out  = flip_v(img_out)
        mask_out = flip_v(mask_out)
        tags.append("flip_v")

    if do_translate:
        img_out  = translate(img_out,  tx, ty)
        mask_out = translate(mask_out, tx, ty)
        tags.append(f"trans_{tx}_{ty}")

    if do_shear:
        img_out  = shear(img_out,  shear_f, shear_axis)
        mask_out = shear(mask_out, shear_f, shear_axis)
        tags.append(f"shear_{shear_axis}_{shear_f:.2f}")

    if do_elastic:
        # Para elástico precisamos de deslocamentos aleatórios; usamos rs
        # diretamente — tanto img quanto mask usarão os MESMOS campos dx/dy
        # porque geramos os campos ANTES de aplicar e passamos o rs (que já
        # foi avançado igualmente por ambos os lados, pois fields são gerados
        # uma única vez dentro da função com rs).
        # Para garantir isso, geramos os campos aqui e aplicamos manualmente.
        dx = gaussian_filter((rs.rand(*mask.shape[:2]) * 2 - 1), sigma_e) * alpha_e
        dy = gaussian_filter((rs.rand(*mask.shape[:2]) * 2 - 1), sigma_e) * alpha_e

        xg, yg = np.meshgrid(np.arange(w), np.arange(h))
        indices = (yg + dy).reshape(-1), (xg + dx).reshape(-1)

        # Máscara
        mask_def = map_coordinates(mask_out, indices, order=1, mode='constant', cval=0)
        mask_out = mask_def.reshape(mask.shape[:2]).astype(mask.dtype)

        # Imagem
        if img_out.ndim == 2:
            img_def = map_coordinates(img_out, indices, order=1, mode='constant', cval=0)
            img_out = img_def.reshape(img_out.shape[:2]).astype(img_out.dtype)
        else:
            channels = []
            for c in range(img_out.shape[2]):
                ch = map_coordinates(img_out[:, :, c], indices, order=1,
                                     mode='constant', cval=0)
                channels.append(ch.reshape(img_out.shape[:2]))
            img_out = np.stack(channels, axis=2).astype(img_out.dtype)

        tags.append(f"elastic_{alpha_e:.1f}_{sigma_e:.1f}")

    # Garante máscara binária (0 / 255)
    _, mask_out = cv2.threshold(mask_out.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    description = "_".join(tags) if tags else "identity"
    return img_out, mask_out, description


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def generate_pairs(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    count: int = 1600,
    seed: int = 42,
    include_elastic: bool = True,
) -> None:
    images_path = Path(images_dir)
    masks_path  = Path(masks_dir)
    out_path    = Path(output_dir)

    out_images = out_path / "images"
    out_masks  = out_path / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # ---- Descobre arquivos originais ---------------------------------------
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    all_imgs = sorted([f for f in images_path.iterdir()
                       if f.suffix.lower() in img_exts])

    if not all_imgs:
        print(f"[ERRO] Nenhuma imagem encontrada em: {images_path}")
        sys.exit(1)

    print(f"Imagens originais encontradas : {len(all_imgs)}")
    print(f"Pares a gerar                 : {count}")
    print(f"Saída                         : {out_path}")
    print(f"Semente aleatória             : {seed}")
    print(f"Deformação elástica           : {'sim' if include_elastic else 'não'}")
    print()

    rng = np.random.RandomState(seed)

    # Amostragem com reposição (permite gerar mais pares do que originais)
    chosen_indices = rng.randint(0, len(all_imgs), size=count)

    # ---- Log CSV -----------------------------------------------------------
    log_path = out_path / "pairs_log.csv"
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "index", "output_image", "output_mask",
        "source_image", "source_mask", "transformations"
    ])

    # ---- Loop de geração ---------------------------------------------------
    generated = 0
    errors    = 0

    for i, src_idx in enumerate(tqdm(chosen_indices, desc="Gerando pares")):
        img_file  = all_imgs[src_idx]
        mask_file = masks_path / img_file.name   # mesmo nome de arquivo

        # Tenta encontrar a máscara com mesma raiz mas extensão diferente
        if not mask_file.exists():
            stem = img_file.stem
            found = list(masks_path.glob(f"{stem}.*"))
            if not found:
                tqdm.write(f"[AVISO] Máscara não encontrada para {img_file.name}, pulando.")
                errors += 1
                continue
            mask_file = found[0]

        # Carrega
        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            tqdm.write(f"[AVISO] Falha ao ler {img_file.name}, pulando.")
            errors += 1
            continue

        # Garante máscara binária
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Aplica transformação
        img_t, mask_t, desc = apply_random_transform_pair(
            image, mask, rng, include_elastic=include_elastic
        )

        # Nomes de saída
        out_img_name  = f"pair_{i:04d}.png"
        out_mask_name = f"pair_{i:04d}.png"

        cv2.imwrite(str(out_images / out_img_name),  img_t)
        cv2.imwrite(str(out_masks  / out_mask_name), mask_t)

        csv_writer.writerow([
            i,
            f"images/{out_img_name}",
            f"masks/{out_mask_name}",
            img_file.name,
            mask_file.name,
            desc,
        ])

        generated += 1

    csv_file.close()

    # ---- Resumo ------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"✓ Pares gerados com sucesso : {generated}")
    if errors:
        print(f"⚠  Pares ignorados (erro)   : {errors}")
    print(f"✓ Imagens salvas em         : {out_images}")
    print(f"✓ Máscaras salvas em        : {out_masks}")
    print(f"✓ Log CSV salvo em          : {log_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera pares (imagem + máscara) com transformações geométricas idênticas."
    )
    parser.add_argument(
        "--images",
        default=r"D:\dataset\tgs-salt\train\images",
        help="Diretório com as imagens originais",
    )
    parser.add_argument(
        "--masks",
        default=r"D:\dataset\tgs-salt\train\masks",
        help="Diretório com as máscaras originais",
    )
    parser.add_argument(
        "--output",
        default=r"output\pairs1600",
        help="Diretório de saída dos pares gerados",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1600,
        help="Número de pares a gerar (default: 1600)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória (default: 42)",
    )
    parser.add_argument(
        "--no-elastic",
        action="store_true",
        help="Desativa a deformação elástica",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pairs(
        images_dir    = args.images,
        masks_dir     = args.masks,
        output_dir    = args.output,
        count         = args.count,
        seed          = args.seed,
        include_elastic = not args.no_elastic,
    )
