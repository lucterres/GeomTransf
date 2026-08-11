"""
generate_pairs_seismic.py
--------------------------
Gera 1600 pares (imagem + máscara) com augmentações físicamente adequadas
para dados sísmicos, seguindo as recomendações da literatura:

  ✓ Flip horizontal leve          — simetria lateral aceitável em sísmica
  ✓ Rotação muito leve (±10°)     — pequenas inclinações estruturais
  ✓ Translação suave (±5%)        — shift espacial realista
  ✓ Escala discreta (0.90–1.10×)  — variação sutil de escala
  ✓ Ruído Gaussiano controlado    — simula ruído sísmico (SOMENTE na imagem)
  ✗ Flip vertical                 — altera polaridade física → desativado
  ✗ Deformação elástica agressiva — apaga falhas/horizontes → desativado
  ✗ Shear forte                   — distorção não-física    → desativado
  ✗ Rotação grande                — altera geometria estrutural → evitado

Saída:
    output/pairs1600_seismic/images/     → imagens augmentadas
    output/pairs1600_seismic/masks/      → máscaras com mesma transformação
    output/pairs1600_seismic/pairs_log.csv

Uso:
    python scripts/generate_pairs_seismic.py [opções]

Opções:
    --images   DIR   Imagens originais   (default: D:\\dataset\\tgs-salt\\train\\images)
    --masks    DIR   Máscaras originais  (default: D:\\dataset\\tgs-salt\\train\\masks)
    --output   DIR   Diretório de saída  (default: output\\pairs1600_seismic)
    --count    N     Número de pares     (default: 1600)
    --seed     N     Semente aleatória   (default: 99)
    --no-noise       Desativa ruído Gaussiano
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Transformações geométricas leves (para imagem e máscara)
# ---------------------------------------------------------------------------

def rotate_light(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotação leve. Usa INTER_NEAREST para máscara, INTER_LINEAR para imagem."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp,
                          borderMode=cv2.BORDER_REFLECT_101)


def scale_discrete(img: np.ndarray, factor: float) -> np.ndarray:
    """Escala discreta centrada."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, factor)
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp,
                          borderMode=cv2.BORDER_REFLECT_101)


def translate_soft(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Translação suave com padding por reflexão."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp,
                          borderMode=cv2.BORDER_REFLECT_101)


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


# ---------------------------------------------------------------------------
# Ruído Gaussiano controlado (aplicado APENAS à imagem, não à máscara)
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: np.ndarray, sigma: float, rs: np.random.RandomState) -> np.ndarray:
    """
    Adiciona ruído Gaussiano à imagem com desvio padrão `sigma`.
    Respeita o dtype original (uint8) clampando [0, 255].
    """
    noise = rs.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Augmentação de um par (imagem, máscara) — mesmos parâmetros geométricos
# ---------------------------------------------------------------------------

def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rs: np.random.RandomState,
    add_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Aplica augmentações sísmicas leves ao par (imagem, máscara).

    Parâmetros geométricos são sorteados UMA VEZ e aplicados igualmente
    a imagem e máscara. O ruído Gaussiano é aplicado SOMENTE à imagem.

    Returns:
        (imagem_aug, mascara_aug, descricao)
    """

    # ── Sorteia TODOS os parâmetros antes de qualquer aplicação ──────────────

    # Flip horizontal (50%)
    do_flip   = rs.rand() < 0.50

    # Rotação leve: ±10°  (70%)
    do_rotate = rs.rand() < 0.70
    angle     = rs.uniform(-10.0, 10.0)

    # Translação suave: ±5% da dimensão  (60%)
    do_trans  = rs.rand() < 0.60
    h, w      = mask.shape[:2]
    tx        = int(rs.uniform(-w * 0.05, w * 0.05))
    ty        = int(rs.uniform(-h * 0.05, h * 0.05))

    # Escala discreta: 0.90–1.10  (50%)
    do_scale  = rs.rand() < 0.50
    scale_f   = rs.uniform(0.90, 1.10)

    # Ruído Gaussiano na imagem: σ ∈ [5, 20]  (60%)
    do_noise  = add_noise and (rs.rand() < 0.60)
    noise_sig = rs.uniform(5.0, 20.0)

    # ── Aplica transformações geométricas (iguais a imagem e máscara) ────────
    img_out  = image.copy()
    mask_out = mask.copy()
    tags: List[str] = []

    if do_flip:
        img_out  = flip_horizontal(img_out)
        mask_out = flip_horizontal(mask_out)
        tags.append("flip_h")

    if do_rotate:
        img_out  = rotate_light(img_out,  angle)
        mask_out = rotate_light(mask_out, angle)
        tags.append(f"rot_{angle:.1f}")

    if do_trans:
        img_out  = translate_soft(img_out,  tx, ty)
        mask_out = translate_soft(mask_out, tx, ty)
        tags.append(f"trans_{tx}_{ty}")

    if do_scale:
        img_out  = scale_discrete(img_out,  scale_f)
        mask_out = scale_discrete(mask_out, scale_f)
        tags.append(f"scale_{scale_f:.2f}")

    # ── Ruído Gaussiano — SOMENTE na imagem ──────────────────────────────────
    if do_noise:
        img_out = add_gaussian_noise(img_out, noise_sig, rs)
        tags.append(f"noise_sigma{noise_sig:.1f}")

    # Garante máscara binária
    _, mask_out = cv2.threshold(mask_out.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    description = "_".join(tags) if tags else "identity"
    return img_out, mask_out, description


# ---------------------------------------------------------------------------
# Geração principal
# ---------------------------------------------------------------------------

def generate_pairs(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    count: int = 1600,
    seed: int = 99,
    add_noise: bool = True,
) -> None:
    images_path = Path(images_dir)
    masks_path  = Path(masks_dir)
    out_path    = Path(output_dir)

    out_images = out_path / "images"
    out_masks  = out_path / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # ── Descobre pares disponíveis ───────────────────────────────────────────
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
    print(f"Ruído Gaussiano               : {'sim' if add_noise else 'não'}")
    print(f"Augmentações sísmicas leves   : flip_h | rot ±10° | trans ±5% | scale 0.90–1.10")
    print()

    rng = np.random.RandomState(seed)

    # Amostragem sem reposição se count ≤ disponíveis, senão com reposição
    replace = count > len(all_imgs)
    chosen_indices = rng.choice(len(all_imgs), size=count, replace=replace)

    # ── Log CSV ──────────────────────────────────────────────────────────────
    log_path = out_path / "pairs_log.csv"
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "index", "output_image", "output_mask",
        "source_image", "source_mask", "augmentations"
    ])

    # ── Loop de geração ──────────────────────────────────────────────────────
    generated = 0
    errors    = 0

    for i, src_idx in enumerate(tqdm(chosen_indices, desc="Gerando pares sísmicos")):
        img_file  = all_imgs[src_idx]
        mask_file = masks_path / img_file.name

        if not mask_file.exists():
            stem  = img_file.stem
            found = list(masks_path.glob(f"{stem}.*"))
            if not found:
                tqdm.write(f"[AVISO] Máscara não encontrada para {img_file.name}, pulando.")
                errors += 1
                continue
            mask_file = found[0]

        image = cv2.imread(str(img_file),  cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            tqdm.write(f"[AVISO] Falha ao ler {img_file.name}, pulando.")
            errors += 1
            continue

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        img_aug, mask_aug, desc = augment_pair(image, mask, rng, add_noise=add_noise)

        out_img_name  = f"seismic_{i:04d}.png"
        out_mask_name = f"seismic_{i:04d}.png"

        cv2.imwrite(str(out_images / out_img_name),  img_aug)
        cv2.imwrite(str(out_masks  / out_mask_name), mask_aug)

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

    # ── Resumo ───────────────────────────────────────────────────────────────
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
# Função de completar slots faltantes
# ---------------------------------------------------------------------------

def resume_pairs(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    count: int = 1600,
    seed: int = 200,
    add_noise: bool = True,
) -> None:
    """
    Completa os slots ausentes em output_dir/images sem re-gerar os que já existem.

    - Detecta quais índices seismic_XXXX.png estão faltantes em disco.
    - Sorteia (com reposição) novas imagens originais para cobrir esses slots.
    - Acrescenta as novas linhas ao pairs_log.csv existente.
    """
    images_path = Path(images_dir)
    masks_path  = Path(masks_dir)
    out_path    = Path(output_dir)
    out_images  = out_path / "images"
    out_masks   = out_path / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # ── Determina slots faltantes ─────────────────────────────────────────
    existing = {f.stem for f in out_images.iterdir() if f.suffix == ".png"}
    all_slots = {f"seismic_{i:04d}" for i in range(count)}
    missing_stems = sorted(all_slots - existing)
    missing_indices = [int(s.split("_")[1]) for s in missing_stems]

    if not missing_indices:
        print("Nenhum slot faltante encontrado — dataset já completo.")
        return

    print(f"Slots faltantes detectados    : {len(missing_indices)}")
    print(f"Semente aleatória (resume)    : {seed}")

    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    all_imgs = sorted([f for f in images_path.iterdir()
                       if f.suffix.lower() in img_exts])

    rng = np.random.RandomState(seed)
    # Sorteia com reposição para garantir que preenchemos exatamente os slots
    chosen_indices = rng.choice(len(all_imgs), size=len(missing_indices), replace=True)

    # ── Abre CSV em modo append ───────────────────────────────────────────
    log_path = out_path / "pairs_log.csv"
    csv_file = open(log_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    generated = 0
    errors    = 0

    for slot_idx, src_img_idx in zip(
        tqdm(missing_indices, desc="Completando pares faltantes"), chosen_indices
    ):
        img_file  = all_imgs[src_img_idx]
        mask_file = masks_path / img_file.name

        if not mask_file.exists():
            found = list(masks_path.glob(f"{img_file.stem}.*"))
            if not found:
                tqdm.write(f"[AVISO] Máscara não encontrada para {img_file.name}, pulando.")
                errors += 1
                continue
            mask_file = found[0]

        image = cv2.imread(str(img_file),  cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            tqdm.write(f"[AVISO] Falha ao ler {img_file.name}, pulando.")
            errors += 1
            continue

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        img_aug, mask_aug, desc = augment_pair(image, mask, rng, add_noise=add_noise)

        out_img_name  = f"seismic_{slot_idx:04d}.png"
        out_mask_name = f"seismic_{slot_idx:04d}.png"

        cv2.imwrite(str(out_images / out_img_name),  img_aug)
        cv2.imwrite(str(out_masks  / out_mask_name), mask_aug)

        csv_writer.writerow([
            slot_idx,
            f"images/{out_img_name}",
            f"masks/{out_mask_name}",
            img_file.name,
            mask_file.name,
            desc,
        ])
        generated += 1

    csv_file.close()

    print()
    print("=" * 60)
    print(f"✓ Pares gerados (resume)    : {generated}")
    if errors:
        print(f"⚠  Pares ignorados (erro)   : {errors}")
    total_disk = len(list(out_images.glob("*.png")))
    print(f"✓ Total em disco agora      : {total_disk}")
    print(f"✓ Log CSV atualizado em     : {log_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmentações sísmicas leves: gera pares (imagem + máscara) físicamente consistentes."
    )
    parser.add_argument("--images",    default=r"D:\dataset\tgs-salt\train\images")
    parser.add_argument("--masks",     default=r"D:\dataset\tgs-salt\train\masks")
    parser.add_argument("--output",    default=r"output\pairs1600_seismic")
    parser.add_argument("--count",     type=int, default=1600)
    parser.add_argument("--seed",      type=int, default=99)
    parser.add_argument("--no-noise",  action="store_true",
                        help="Desativa ruído Gaussiano na imagem")
    parser.add_argument("--resume",    action="store_true",
                        help="Completa slots faltantes sem re-gerar os existentes")
    parser.add_argument("--resume-seed", type=int, default=200,
                        help="Semente para o modo resume (default: 200)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        resume_pairs(
            images_dir = args.images,
            masks_dir  = args.masks,
            output_dir = args.output,
            count      = args.count,
            seed       = args.resume_seed,
            add_noise  = not args.no_noise,
        )
    else:
        generate_pairs(
            images_dir = args.images,
            masks_dir  = args.masks,
            output_dir = args.output,
            count      = args.count,
            seed       = args.seed,
            add_noise  = not args.no_noise,
        )
