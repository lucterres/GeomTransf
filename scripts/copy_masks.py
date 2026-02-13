#!/usr/bin/env python3
"""Copy mask PNG files listed in a CSV to a destination folder.

Reads the first column from the CSV (header `id`) and copies files
named <id>.png from the source directory to the destination.

Defaults match your request but can be overridden with CLI args.
"""

from pathlib import Path
import argparse
import csv
import shutil
import sys


def load_ids_from_csv(csv_path: Path):
    ids = []
    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            # Skip header if present
            if i == 0 and row[0].lower() in ('id', 'image_id', 'filename'):
                continue
            ids.append(row[0].strip())
    return ids


def copy_masks(ids, src_dir: Path, dst_dir: Path, dry_run=False, overwrite=False):
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    for ident in ids:
        if not ident:
            continue
        filename = ident if ident.lower().endswith('.png') else f"{ident}.png"
        src = src_dir / filename
        dst = dst_dir / filename

        if not src.exists():
            missing.append(str(src))
            continue

        if dst.exists() and not overwrite:
            # skip existing
            continue

        if dry_run:
            print(f"[DRY] Copy: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def main():
    parser = argparse.ArgumentParser(description="Copy masks listed in CSV to destination folder")
    parser.add_argument('--csv', default='data/saltMaskOk.csv', help='Path to CSV with mask ids (default: data/saltMaskOk.csv)')
    parser.add_argument('--source', default=r'D:\dataset\tgs-salt\train\masks', help='Source masks directory')
    parser.add_argument('--dest', default=r'D:\dataset\tgs-salt\train\masks1090', help='Destination directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be copied without performing copy')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite files in destination if they exist')

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        sys.exit(2)

    ids = load_ids_from_csv(csv_path)
    print(f"Loaded {len(ids)} ids from {csv_path}")

    src_dir = Path(args.source)
    dst_dir = Path(args.dest)

    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        sys.exit(2)

    copied, missing = copy_masks(ids, src_dir, dst_dir, dry_run=args.dry_run, overwrite=args.overwrite)

    print(f"\nDone. Copied: {copied}")
    if missing:
        print(f"Missing files: {len(missing)} (examples):")
        for m in missing[:10]:
            print(f"  {m}")
        print("...")


if __name__ == '__main__':
    main()
