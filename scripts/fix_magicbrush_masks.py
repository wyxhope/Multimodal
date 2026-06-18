#!/usr/bin/env python
"""Invert existing MagicBrush masks so white means edit region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("dataset/magicbrush/metadata.json"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/magicbrush"))
    parser.add_argument("--in-place", action="store_true", help="Overwrite masks and metadata.")
    args = parser.parse_args()

    records = json.loads(args.metadata.read_text(encoding="utf-8"))
    for record in records:
        value = record.get("target_mask") or record.get("mask") or record.get("mask_path")
        if not value:
            continue
        path = Path(value)
        path = path if path.is_absolute() else args.image_root / path
        mask = Image.open(path).convert("L")
        out_path = path if args.in_place else path.with_name(path.stem + "_white_edit.png")
        ImageOps.invert(mask).save(out_path)
        if args.in_place:
            record["mask_semantics"] = "white_edit"
        else:
            record["target_mask"] = str(out_path.relative_to(args.image_root))
            record["mask_semantics"] = "white_edit"

    out_metadata = args.metadata if args.in_place else args.metadata.with_name("metadata_white_edit.json")
    out_metadata.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {out_metadata}")


if __name__ == "__main__":
    main()
