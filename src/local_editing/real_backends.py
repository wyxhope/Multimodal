"""Real-model adapters for GroundingDINO, SAM, and diffusion inpainting."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from dotenv import load_dotenv


@dataclass(frozen=True)
class GroundedMask:
    bbox: tuple[int, int, int, int]
    mask: Image.Image
    label: str
    score: float


class OptionalDependencyError(RuntimeError):
    pass


def load_environment() -> None:
    load_dotenv()
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_env.exists():
        load_dotenv(repo_env, override=False)


def require_package(import_name: str, install_hint: str) -> None:
    try:
        __import__(import_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"Missing optional package {import_name!r}. Install with: {install_hint}"
        ) from exc


def get_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    require_package("torch", "pip install torch")
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class GroundingSamAdapter:
    """Ground a text phrase with GroundingDINO and segment it with SAM."""

    def __init__(
        self,
        grounding_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_checkpoint: str | Path | None = None,
        sam_model_type: str = "vit_h",
        device: str | None = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> None:
        load_environment()
        require_package("torch", "pip install torch")
        require_package("transformers", "pip install transformers")
        require_package("segment_anything", "pip install git+https://github.com/facebookresearch/segment-anything.git")

        import torch
        from segment_anything import SamPredictor, sam_model_registry
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        checkpoint = sam_checkpoint or os.environ.get("SAM_CHECKPOINT")
        if not checkpoint:
            raise OptionalDependencyError(
                "SAM checkpoint is required. Download a SAM checkpoint and set "
                "SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth, or pass --sam-checkpoint."
            )

        self.device = get_device(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        self.detector.eval()

        sam = sam_model_registry[sam_model_type](checkpoint=str(checkpoint))
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        self.torch = torch

    def predict(self, image: Image.Image, object_phrase: str) -> GroundedMask:
        image = image.convert("RGB")
        text = _grounding_text(object_phrase)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.detector(**inputs)

        result = self._post_process_grounding(outputs, inputs, image)
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")
        if boxes is None or len(boxes) == 0:
            raise RuntimeError(f"GroundingDINO found no object for phrase: {object_phrase!r}")

        best_idx = int(scores.argmax().item()) if hasattr(scores, "argmax") else int(np.argmax(scores))
        bbox = _clip_box(_to_list(boxes[best_idx]), image.size)
        score = float(scores[best_idx].item() if hasattr(scores[best_idx], "item") else scores[best_idx])
        label = str(labels[best_idx]) if labels is not None and len(labels) else object_phrase

        self.sam_predictor.set_image(np.array(image))
        masks, mask_scores, _ = self.sam_predictor.predict(
            box=np.array(bbox, dtype=np.float32),
            multimask_output=True,
        )
        mask_idx = int(np.argmax(mask_scores))
        mask = Image.fromarray((masks[mask_idx].astype(np.uint8) * 255), mode="L")
        return GroundedMask(bbox=bbox, mask=mask, label=label, score=score)

    def _post_process_grounding(self, outputs: Any, inputs: Any, image: Image.Image) -> dict[str, Any]:
        target_sizes = [image.size[::-1]]
        method = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(method).parameters
        kwargs: dict[str, Any] = {"target_sizes": target_sizes}
        if "input_ids" in parameters:
            kwargs["input_ids"] = inputs.input_ids
        if "box_threshold" in parameters:
            kwargs["box_threshold"] = self.box_threshold
        elif "threshold" in parameters:
            kwargs["threshold"] = self.box_threshold
        if "text_threshold" in parameters:
            kwargs["text_threshold"] = self.text_threshold
        processed = method(outputs, **kwargs)
        return processed[0]


class StableDiffusionInpaintAdapter:
    """Diffusers-based inpainting backend."""

    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        device: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        load_environment()
        require_package("torch", "pip install torch")
        require_package("diffusers", "pip install diffusers transformers accelerate safetensors")
        require_package("transformers", "pip install transformers")

        import torch
        from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline

        self.device = get_device(device)
        if torch_dtype:
            dtype = getattr(torch, torch_dtype)
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = _load_diffusers_pipeline(
            AutoPipelineForInpainting,
            model_id=model_id,
            torch_dtype=dtype,
            fallback_cls=StableDiffusionInpaintPipeline,
            prefer_fp16_variant=self.device == "cuda",
        )

        self.pipe = self.pipe.to(self.device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        self.torch = torch

    def edit(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, extra objects, text, watermark",
        num_inference_steps: int = 35,
        guidance_scale: float = 7.5,
        strength: float | None = None,
        seed: int | None = 1234,
        size: int | None = 512,
    ) -> Image.Image:
        original_size = image.size
        image = image.convert("RGB")
        mask = mask.convert("L")
        if size:
            image = _resize_square(image, size)
            mask = _resize_square(mask, size)

        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(seed)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if strength is not None:
            kwargs["strength"] = strength

        result = self.pipe(**kwargs).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        return result


class InstructPix2PixAdapter:
    """Mask-free instruction-editing baseline."""

    def __init__(
        self,
        model_id: str = "timbrooks/instruct-pix2pix",
        device: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        load_environment()
        require_package("torch", "pip install torch")
        require_package("diffusers", "pip install diffusers transformers accelerate safetensors")
        require_package("transformers", "pip install transformers")

        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self.device = get_device(device)
        if torch_dtype:
            dtype = getattr(torch, torch_dtype)
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = _load_diffusers_pipeline(
            StableDiffusionInstructPix2PixPipeline,
            model_id=model_id,
            torch_dtype=dtype,
            prefer_fp16_variant=self.device == "cuda",
        )
        self.pipe = self.pipe.to(self.device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        self.torch = torch

    def edit(
        self,
        image: Image.Image,
        instruction: str,
        num_inference_steps: int = 35,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        seed: int | None = 1234,
        size: int | None = 512,
    ) -> Image.Image:
        original_size = image.size
        image = image.convert("RGB")
        if size:
            image = _resize_square(image, size)

        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=instruction,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        return result


def _grounding_text(object_phrase: str) -> str:
    text = object_phrase.strip().lower()
    return text if text.endswith(".") else f"{text}."


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [float(x) for x in value]


def _clip_box(box: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    return x0, y0, x1, y1


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _load_diffusers_pipeline(
    pipeline_cls: Any,
    model_id: str,
    torch_dtype: Any,
    fallback_cls: Any | None = None,
    prefer_fp16_variant: bool = False,
) -> Any:
    attempts: list[dict[str, Any]] = []
    if prefer_fp16_variant:
        attempts.append({"torch_dtype": torch_dtype, "use_safetensors": True, "variant": "fp16"})
    attempts.append({"torch_dtype": torch_dtype, "use_safetensors": True})
    attempts.append({"torch_dtype": torch_dtype})

    classes = [pipeline_cls]
    if fallback_cls is not None and fallback_cls is not pipeline_cls:
        classes.append(fallback_cls)

    errors: list[str] = []
    for cls in classes:
        for kwargs in attempts:
            try:
                return cls.from_pretrained(model_id, **kwargs)
            except Exception as exc:
                errors.append(f"{cls.__name__} with {kwargs}: {type(exc).__name__}: {exc}")

    joined = "\n\n".join(errors[-4:])
    raise RuntimeError(f"Could not load diffusers pipeline {model_id!r}. Last errors:\n{joined}")
