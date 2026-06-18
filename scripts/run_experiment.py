#!/usr/bin/env python
"""Run real-model local-editing experiments on a dataset metadata file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from PIL import Image

from local_editing.masking import select_real_mask
from local_editing.masking import invert_mask, resize_mask
from local_editing.metrics import evaluate
from local_editing.parser import build_inpaint_prompt, parse_instruction
from local_editing.real_backends import (
    GroundingSamAdapter,
    InstructPix2PixAdapter,
    StableDiffusionInpaintAdapter,
)
from local_editing.visualization import draw_box, save_case_grid


AUTO_MASK_STRATEGIES = {"box", "sam", "dilated_sam"}
MASK_STRATEGIES = {"box", "sam", "dilated_sam", "oracle", "dilated_oracle"}
ALL_STRATEGIES = ("mask_free", "box", "sam", "dilated_sam", "oracle", "dilated_oracle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True, help="JSON/JSONL/CSV with image and instruction fields.")
    parser.add_argument("--image-root", type=Path, default=None, help="Root directory for relative image paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=list(ALL_STRATEGIES))
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--dilation-radius", type=int, default=8)

    parser.add_argument("--inpaint-model-id", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--instructpix2pix-model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    records = load_records(args.metadata)
    if args.limit is not None:
        records = records[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root or args.metadata.parent
    needs_grounding = any(strategy in AUTO_MASK_STRATEGIES for strategy in args.strategies)
    needs_inpaint = any(strategy in MASK_STRATEGIES for strategy in args.strategies)
    needs_mask_free = "mask_free" in args.strategies

    segmenter = (
        GroundingSamAdapter(
            grounding_model_id=args.grounding_model_id,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        if needs_grounding
        else None
    )
    inpainter = (
        StableDiffusionInpaintAdapter(model_id=args.inpaint_model_id, device=args.device)
        if needs_inpaint
        else None
    )
    pix2pix = (
        InstructPix2PixAdapter(model_id=args.instructpix2pix_model_id, device=args.device)
        if needs_mask_free
        else None
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        try:
            case_id = record_id(record, index)
            image_path = resolve_image_path(record, image_root)
            instruction_text = record["instruction"]
            original = Image.open(image_path).convert("RGB")
            instruction = parse_instruction(instruction_text)
            prompt = build_inpaint_prompt(instruction, original_text=instruction_text)
            oracle_mask = load_optional_mask(record, image_root, original.size)

            case_dir = args.output_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)

            grounded = None
            grounding_error: Exception | None = None
            if needs_grounding:
                try:
                    if segmenter is None:
                        raise RuntimeError("Grounding/SAM adapter was not initialized")
                    grounded = segmenter.predict(original, instruction.target)
                    grounded.mask.save(case_dir / "sam_mask.png")
                    draw_box(original, grounded.bbox).save(case_dir / "grounding_box.png")
                except Exception as exc:
                    grounding_error = exc

            for strategy in args.strategies:
                try:
                    if strategy == "mask_free":
                        if pix2pix is None:
                            raise RuntimeError("InstructPix2Pix adapter was not initialized")
                        edit_mask = None
                        edited = pix2pix.edit(
                            image=original,
                            instruction=instruction_text,
                            num_inference_steps=args.num_inference_steps,
                            image_guidance_scale=args.image_guidance_scale,
                            guidance_scale=args.guidance_scale,
                            seed=args.seed,
                            size=args.size,
                        )
                    else:
                        if inpainter is None:
                            raise RuntimeError("Inpainting adapter was not initialized")
                        if strategy in AUTO_MASK_STRATEGIES:
                            if grounding_error is not None:
                                raise RuntimeError(f"Grounding/SAM failed: {grounding_error}") from grounding_error
                            if grounded is None:
                                raise RuntimeError("Grounding/SAM result is missing")
                            sam_mask = grounded.mask
                            bbox = grounded.bbox
                        else:
                            if oracle_mask is None:
                                raise RuntimeError(f"{strategy} requires target_mask/mask_path in metadata")
                            sam_mask = oracle_mask
                            bbox = (0, 0, original.size[0], original.size[1])

                        edit_mask = select_real_mask(
                            strategy=strategy,
                            image=original,
                            sam_mask=sam_mask,
                            bbox=bbox,
                            dilation_radius=args.dilation_radius,
                            oracle_mask=oracle_mask,
                        )
                        edited = inpainter.edit(
                            image=original,
                            mask=edit_mask,
                            prompt=prompt,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            seed=args.seed,
                            size=args.size,
                        )

                    edited_path = case_dir / f"{strategy}_edited.png"
                    grid_path = case_dir / f"{strategy}_grid.png"
                    edited.save(edited_path)
                    bbox_image = draw_box(original, grounded.bbox) if grounded is not None else original
                    display_mask = edit_mask or oracle_mask or (grounded.mask if grounded is not None else None)
                    if display_mask is None:
                        display_mask = Image.new("L", original.size, 255)
                    save_case_grid(
                        original=original,
                        mask=display_mask,
                        bbox_image=bbox_image,
                        edited=edited,
                        output_path=grid_path,
                        title=f"{strategy}: {instruction_text}",
                    )

                    eval_mask = oracle_mask or edit_mask or (grounded.mask if grounded is not None else display_mask)
                    scores = evaluate(original, edited, eval_mask)
                    row = {
                        "case_id": case_id,
                        "strategy": strategy,
                        "image": str(image_path),
                        "instruction": instruction_text,
                        "operation": instruction.operation,
                        "target": instruction.target,
                        "replacement": instruction.replacement or "",
                        "target_color": instruction.target_color or "",
                        "prompt": prompt,
                        "grounded_label": grounded.label if grounded is not None else "",
                        "grounded_score": grounded.score if grounded is not None else "",
                        "has_oracle_mask": oracle_mask is not None,
                        "bbox": json.dumps(list(grounded.bbox)) if grounded is not None else "",
                        "edited_image": str(edited_path),
                        "grid": str(grid_path),
                        **scores,
                    }
                    metric_rows.append(row)
                    prediction_rows.append(row)
                except Exception as exc:
                    failures.append(
                        {
                            "index": index,
                            "case_id": case_id,
                            "strategy": strategy,
                            "record": record,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
        except Exception as exc:
            failures.append(
                {
                    "index": index,
                    "record": record,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    write_csv(metric_rows, args.output_dir / "metrics.csv")
    write_summary(metric_rows, args.output_dir / "summary.csv")
    (args.output_dir / "predictions.json").write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
    (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Finished {len(metric_rows)} strategy outputs with {len(failures)} failed records.")


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data["data"]
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported metadata format: {path}")


def resolve_image_path(record: dict[str, Any], image_root: Path) -> Path:
    value = record.get("image") or record.get("source_image") or record.get("image_path")
    if not value:
        raise ValueError(f"Record is missing image/source_image/image_path: {record}")
    path = Path(value)
    return path if path.is_absolute() else image_root / path


def load_optional_mask(
    record: dict[str, Any],
    image_root: Path,
    image_size: tuple[int, int] | None = None,
) -> Image.Image | None:
    value = record.get("target_mask") or record.get("mask") or record.get("mask_path")
    if not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else image_root / path
    mask = Image.open(path).convert("L")
    semantics = str(record.get("mask_semantics", "white_edit")).lower()
    if semantics in {"white_keep", "keep_white", "raw_magicbrush", "black_edit"}:
        mask = invert_mask(mask)
    if image_size is not None:
        mask = resize_mask(mask, image_size)
    return mask


def record_id(record: dict[str, Any], index: int) -> str:
    raw = record.get("id") or record.get("case_id") or record.get("uid") or f"case_{index:04d}"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(raw))


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    metrics = [
        "change_inside",
        "change_outside",
        "locality_score",
        "background_mse",
        "background_psnr",
        "background_ssim",
    ]
    strategies = sorted({str(row["strategy"]) for row in rows})
    with output_path.open("w", encoding="utf-8") as f:
        f.write("strategy," + ",".join(metrics) + "\n")
        for strategy in strategies:
            subset = [row for row in rows if row["strategy"] == strategy]
            values = [statistics.mean(float(row[m]) for row in subset) for m in metrics]
            f.write(strategy + "," + ",".join(f"{v:.6f}" for v in values) + "\n")


if __name__ == "__main__":
    main()
