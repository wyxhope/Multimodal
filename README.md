# Text-Guided Local Image Editing with Object-Aware Masking

This is the final real-model version of the project. The pipeline directly uses:

```text
image + instruction
  -> DeepSeek V4 Flash instruction parser
  -> GroundingDINO text-conditioned detection
  -> SAM box-prompted object mask
  -> Stable Diffusion Inpainting / InstructPix2Pix
  -> locality and background-preservation metrics
```

## Files

- `src/local_editing/parser.py`: calls DeepSeek V4 Flash and returns structured edit fields.
- `src/local_editing/real_backends.py`: wraps GroundingDINO, SAM, Stable Diffusion Inpainting, and InstructPix2Pix.
- `src/local_editing/masking.py`: builds box, SAM, dilated-SAM, and full-image masks.
- `src/local_editing/metrics.py`: computes inside/outside change, locality, background MSE/PSNR/SSIM.
- `src/local_editing/visualization.py`: saves qualitative grids and difference maps.
- `scripts/run_real_edit.py`: runs one real image and one instruction.
- `scripts/run_experiment.py`: runs a batch of real image-editing cases from a metadata file.
- `scripts/prepare_magicbrush_subset.py`: streams a small MagicBrush subset into local image files.

## Setup

```bash
pip install -r requirements.txt
```

Set the DeepSeek API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
SAM_CHECKPOINT=/absolute/path/to/sam_vit_h_4b8939.pth
```

Shell environment variables still work and take priority over `.env`.

The default real models are:

- GroundingDINO: `IDEA-Research/grounding-dino-base`
- SAM: `vit_h`
- Stable Diffusion Inpainting: `stable-diffusion-v1-5/stable-diffusion-inpainting`
- InstructPix2Pix: `timbrooks/instruct-pix2pix`

## Single Image

```bash
PYTHONPATH=src python scripts/run_real_edit.py \
  --image path/to/input.jpg \
  --instruction "change the red cup to a blue cup" \
  --output-dir outputs/single_cup \
  --mask-strategy sam
```

Output files:

- `edited.png`
- `mask.png`
- `box.png`
- `grid.png`
- `metadata.json`

## Supported Tasks

This code is designed for **local edits of an object or region that already exists in the input image**.

Good fits:

- Change an existing object's color, e.g. `change the red cup to a blue cup`.
- Remove an existing object, e.g. `remove the chair`.
- Replace an existing object, e.g. `replace the apple with an orange`.
- Modify a visible object attribute when GroundingDINO can still localize the object, e.g. `make the cow smile`.

Weak or unsupported fits:

- Add a new object at a location where no existing target object can be grounded, e.g. `put the zebras next to a river`.
- Broad scene edits, e.g. `make it sunset`, `change the room into a kitchen`.
- Human identity, pose, hair, or expression edits when the target region is ambiguous.
- Multi-object edits requiring several masks.
- Instructions that require layout planning, e.g. `add a dolphin jumping out of the water`.

The reason is structural: the proposed pipeline needs a text phrase that GroundingDINO can find in the original image. If the instruction asks to add something that is not present, there may be no object to ground and no reliable mask to inpaint.

For MagicBrush, do not expect all instructions to work. Use this project as:

```text
main experiment: existing-object local edits
failure analysis: additions, broad scene edits, ambiguous human edits
```

If you want to cover more MagicBrush tasks, add one of these extensions:

- For object insertion: use an LLM/VLM to predict a placement box, then run inpainting on that box.
- For multi-object edits: parse multiple targets and run GroundingDINO/SAM for each one.
- For global style or scene changes: use InstructPix2Pix or another global editor instead of the local-mask pipeline.
- For ambiguous masks: add a small manual UI to correct the box or mask before inpainting.

## Batch Experiment

Prepare a small MagicBrush subset without downloading/materializing the full dataset:

```bash
PYTHONPATH=src python scripts/prepare_magicbrush_subset.py \
  --output-dir dataset/magicbrush \
  --split dev \
  --limit 50 \
  --hf-cache-dir /root/autodl-tmp/hf_cache
```

Mask convention in this project is **white = edit / repaint, black = preserve**. MagicBrush raw masks are inverted during preparation so that they match diffusers inpainting. If you prepared the dataset with an older script, repair it with:

```bash
PYTHONPATH=src python scripts/fix_magicbrush_masks.py \
  --metadata dataset/magicbrush/metadata.json \
  --image-root dataset/magicbrush \
  --in-place
```

If a previous full `load_dataset("osunlp/MagicBrush")` run filled the root disk, remove the partial Hugging Face cache first:

```bash
rm -rf /root/.cache/huggingface/hub/datasets--osunlp--MagicBrush
rm -rf /root/.cache/huggingface/datasets/osunlp___magic_brush
```

Create a JSON, JSONL, or CSV metadata file. Required fields:

```json
[
  {
    "id": "case_001",
    "image": "images/example.jpg",
    "instruction": "change the red cup to a blue cup"
  }
]
```

Optional mask fields are supported for evaluation:

```json
{
  "target_mask": "masks/example_mask.png"
}
```

If no target mask is provided, the generated SAM mask is used as the evaluation mask.

Run all strategies:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --metadata dataset/metadata.json \
  --image-root dataset \
  --output-dir outputs/experiment \
  --strategies mask_free box sam dilated_sam oracle
```

Strategies:

- `mask_free`: real InstructPix2Pix baseline.
- `box`: Stable Diffusion Inpainting with the GroundingDINO bounding box mask.
- `sam`: Stable Diffusion Inpainting with the SAM object mask.
- `dilated_sam`: Stable Diffusion Inpainting with a slightly expanded SAM mask.
- `oracle`: Stable Diffusion Inpainting with the dataset-provided edit mask, e.g. MagicBrush `mask_img`.
- `dilated_oracle`: Stable Diffusion Inpainting with a dilated dataset-provided edit mask.

The comparison is therefore:

```text
InstructPix2Pix without explicit localization
vs.
GroundingDINO + Stable Diffusion Inpainting using box mask
vs.
GroundingDINO + SAM + Stable Diffusion Inpainting using object mask
vs.
GroundingDINO + SAM + Stable Diffusion Inpainting using dilated object mask
vs.
Stable Diffusion Inpainting using the dataset-provided oracle edit mask
```

So `box`, `sam`, `dilated_sam`, and `oracle` are not unrelated systems. They are mask ablations inside the same localization-plus-inpainting pipeline. `oracle` is not a deployable automatic method; it is an upper-bound experiment that uses MagicBrush's ground-truth edit region.

Batch output:

- `metrics.csv`: per-case metrics.
- `summary.csv`: average metrics per strategy.
- `predictions.json`: parsed instructions, prompts, boxes, scores, and output paths.
- `failures.json`: records that failed during parsing, grounding, segmentation, or generation.
- One subdirectory per case containing edited images and qualitative grids.

## Real Dataset Choices

Yes, you can use real datasets. Good choices for this project are:

- **MagicBrush**: best match for instruction-guided image editing because it provides source images, editing instructions, and target edited images.
- **COCO**: good for manually writing controlled edits around common objects.
- **OpenImages**: useful if you want many object categories, but it needs more filtering.

For this project scope, MagicBrush plus a small manually filtered COCO subset is the most practical combination.

## How MagicBrush Tasks Should Be Handled

MagicBrush contains many open-ended edit types, not only simple object replacement. Use different routes:

| MagicBrush instruction type | Recommended route |
| --- | --- |
| Existing object color/attribute/removal/replacement | `box`, `sam`, `dilated_sam`, and `oracle` |
| Add a new object at a vague location | `oracle` if using MagicBrush masks; otherwise needs a placement-box predictor or user mask |
| Background or region edit | `oracle`; automatic GroundingDINO/SAM may not know what to segment |
| Human expression, hair, clothing, pose | `oracle` or a specialized human/face parser |
| Multi-object edit | needs multi-target parsing and multiple masks |
| Global style/scene edit | `mask_free` or another global editor, not the local-mask pipeline |

For course experiments, the cleanest setup is:

```text
mask_free: global baseline
box/sam/dilated_sam: automatic local editing on cases where the target exists
oracle: MagicBrush upper bound using the dataset mask, including harder insertion/background tasks
failure analysis: cases where automatic grounding cannot find a good target
```

## Startup Commands

Install dependencies:

```bash
cd /root/autodl-tmp/Multimodal
pip install -r requirements.txt
```

For RTX 5090 / 50-series GPUs, install a CUDA 12.8 PyTorch wheel first. The old `torch 2.5.1+cu124` wheel does not support `sm_120` and Transformers also requires `torch>=2.6` for safe model loading:

```bash
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-cu128.txt
pip install -r requirements.txt
```

Verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

Expected for RTX 5090: PyTorch 2.7 or newer, CUDA 12.8, capability `(12, 0)`.

Create and edit the environment file:

```bash
cp .env.example .env
vim .env
```

Download a SAM checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Use this in `.env`:

```text
SAM_CHECKPOINT=/root/autodl-tmp/Multimodal/checkpoints/sam_vit_h_4b8939.pth
```

If an earlier full MagicBrush download filled the root disk, clean it:

```bash
rm -rf /root/.cache/huggingface/hub/datasets--osunlp--MagicBrush
rm -rf /root/.cache/huggingface/datasets/osunlp___magic_brush
```

Prepare a 5-case smoke-test subset:

```bash
PYTHONPATH=src python scripts/prepare_magicbrush_subset.py \
  --output-dir dataset/magicbrush \
  --split dev \
  --limit 5 \
  --hf-cache-dir /root/autodl-tmp/hf_cache
```

Run a 5-case smoke test:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --metadata dataset/magicbrush/metadata.json \
  --image-root dataset/magicbrush \
  --output-dir outputs/magicbrush_5 \
  --strategies mask_free box sam dilated_sam oracle \
  --limit 5
```

Prepare the recommended 50-case subset:

```bash
PYTHONPATH=src python scripts/prepare_magicbrush_subset.py \
  --output-dir dataset/magicbrush \
  --split dev \
  --limit 50 \
  --hf-cache-dir /root/autodl-tmp/hf_cache
```

Run the main experiment:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --metadata dataset/magicbrush/metadata.json \
  --image-root dataset/magicbrush \
  --output-dir outputs/magicbrush_50 \
  --strategies mask_free box sam dilated_sam oracle
```

Check results:

```bash
cat outputs/magicbrush_50/summary.csv
cat outputs/magicbrush_50/failures.json
```
