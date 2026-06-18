#!/usr/bin/env python
"""Prepare a small MagicBrush subset without materializing the full dataset."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/magicbrush"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("/root/autodl-tmp/hf_cache"))
    parser.add_argument(
        "--keep-raw-mask",
        action="store_true",
        help="Keep MagicBrush mask polarity unchanged. By default masks are inverted so white means edit region.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(args.hf_cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_cache_dir / "datasets"))
    args.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    target_dir = args.output_dir / "targets"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "osunlp/MagicBrush",
        split=args.split,
        streaming=True,
        cache_dir=str(args.hf_cache_dir / "datasets"),
    )

    records: list[dict[str, str]] = []
    for index, item in enumerate(dataset):
        if index >= args.limit:
            break
        case_id = f"magicbrush_{index:04d}"
        source_path = image_dir / f"{case_id}.png"
        mask_path = mask_dir / f"{case_id}_mask.png"
        target_path = target_dir / f"{case_id}_target.png"

        save_image(item["source_img"], source_path)
        save_image(item["mask_img"], mask_path, mode="L", invert=not args.keep_raw_mask)
        if "target_img" in item and item["target_img"] is not None:
            save_image(item["target_img"], target_path)

        record = {
            "id": case_id,
            "image": str(source_path.relative_to(args.output_dir)),
            "target_mask": str(mask_path.relative_to(args.output_dir)),
            "mask_semantics": "white_edit" if not args.keep_raw_mask else "raw_magicbrush",
            "instruction": item["instruction"],
        }
        if target_path.exists():
            record["target_image"] = str(target_path.relative_to(args.output_dir))
        records.append(record)

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} cases to {metadata_path}")


def save_image(value: Any, path: Path, mode: str | None = None, invert: bool = False) -> None:
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict) and value.get("bytes") is not None:
        image = Image.open(BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path") is not None:
        image = Image.open(value["path"])
    else:
        image = Image.open(value)

    if mode:
        image = image.convert(mode)
    else:
        image = image.convert("RGB")
    if invert:
        image = ImageOps.invert(image.convert("L"))
    image.save(path)


if __name__ == "__main__":
    main()
