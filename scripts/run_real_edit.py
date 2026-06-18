#!/usr/bin/env python
"""Run the full real-model local-editing pipeline on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from local_editing.masking import select_real_mask
from local_editing.metrics import evaluate
from local_editing.parser import build_inpaint_prompt, parse_instruction
from local_editing.real_backends import GroundingSamAdapter, StableDiffusionInpaintAdapter
from local_editing.visualization import draw_box, save_case_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--instruction", required=True, help="Natural-language editing instruction.")
    parser.add_argument("--output-dir", type=Path, default=Path("real_outputs"))
    parser.add_argument("--mask-strategy", choices=["box", "sam", "dilated_sam"], default="sam")
    parser.add_argument("--dilation-radius", type=int, default=8)

    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)

    parser.add_argument("--inpaint-model-id", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    instruction = parse_instruction(args.instruction)
    prompt = build_inpaint_prompt(instruction, original_text=args.instruction)

    segmenter = GroundingSamAdapter(
        grounding_model_id=args.grounding_model_id,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    grounded = segmenter.predict(image, instruction.target)

    edit_mask = select_real_mask(
        strategy=args.mask_strategy,
        image=image,
        sam_mask=grounded.mask,
        bbox=grounded.bbox,
        dilation_radius=args.dilation_radius,
    )

    inpainter = StableDiffusionInpaintAdapter(
        model_id=args.inpaint_model_id,
        device=args.device,
    )
    edited = inpainter.edit(
        image=image,
        mask=edit_mask,
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        size=args.size,
    )

    edited_path = args.output_dir / "edited.png"
    mask_path = args.output_dir / "mask.png"
    box_path = args.output_dir / "box.png"
    grid_path = args.output_dir / "grid.png"
    metadata_path = args.output_dir / "metadata.json"

    edited.save(edited_path)
    edit_mask.save(mask_path)
    draw_box(image, grounded.bbox).save(box_path)
    save_case_grid(
        image,
        edit_mask,
        draw_box(image, grounded.bbox),
        edited,
        grid_path,
        args.instruction,
    )

    metadata = {
        "instruction": args.instruction,
        "parsed_instruction": instruction.__dict__,
        "inpaint_prompt": prompt,
        "grounded_label": grounded.label,
        "grounded_score": grounded.score,
        "bbox": list(grounded.bbox),
        "mask_strategy": args.mask_strategy,
        "edited_image": str(edited_path),
        "mask": str(mask_path),
        "grid": str(grid_path),
    }
    try:
        metadata["metrics_against_mask"] = evaluate(image, edited, edit_mask)
    except Exception:
        pass
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
