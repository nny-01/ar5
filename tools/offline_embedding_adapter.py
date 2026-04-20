"""Offline embedding adapter for RGBT + AR training.

Purpose
-------
This script produces *dataset-adapted* text embeddings that are used by the AR
(AlignmentRegion) modules during training. Adaptation is performed fully
offline, BEFORE training starts. Nothing here is run inside the training
forward pass.

Design principle
----------------
The CLIP text embedding is the **semantic anchor**. We perform only a small,
bounded correction toward a dataset-derived visual prototype. In particular:

* We do NOT replace the text embedding by the mean visual feature.
* We do NOT do any unconstrained EMA update during training.
* Instead, for each class ``c`` we compute a visual prototype ``p_c`` by
  running the CLIP image encoder over GT box crops, and build the adapted
  text embedding as

      e_final = normalize((1 - alpha) * e_clip + alpha * p_c)

  with a small ``alpha`` (0.1 for class embeddings, 0.15~0.25 for
  descriptions). This keeps the adapted embedding close to the CLIP text
  manifold while giving it a gentle dataset bias.

Inputs
------
* ``--data`` dataset YAML (YOLO-style, with ``names`` and ``train`` paths).
* ``--desc-rgb`` per-class RGB description embeddings
  (``{'embeddings': Tensor[N,512], 'class_map': Tensor[N]}``).
* ``--desc-ir`` per-class IR description embeddings (same format).
* The CLIP ``ViT-B/32`` model (downloaded via the ``clip`` package).

Outputs (into ``--output-dir``)
-------------------------------
* ``class_adapted_embeddings.pt``         – Tensor[nc, 512]
* ``desc_rgb_adapted_embeddings.pt``      – dict with ``embeddings`` + ``class_map``
* ``desc_ir_adapted_embeddings.pt``       – dict with ``embeddings`` + ``class_map``
* ``prototype_stats.pt`` (optional)       – dict with per-class stats for debugging

Usage
-----
    python tools/offline_embedding_adapter.py \
        --data ultralytics/cfg/datasets/M3FD-rgbt.yaml \
        --desc-rgb prompts/M3FD_prompts/perclass_desc_rgb_embeddings.pt \
        --desc-ir  prompts/M3FD_prompts/perclass_desc_ir_embeddings.pt \
        --output-dir prompts/M3FD_prompts/adapted \
        --alpha-class 0.1 \
        --alpha-desc 0.2 \
        --split train \
        --ir-replace images:images_ir \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("offline_embedding_adapter")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _load_yaml(data_yaml: Path) -> dict:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _resolve_class_names(cfg: dict) -> List[str]:
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    if isinstance(names, list):
        return list(names)
    raise ValueError(f"Unsupported 'names' format in dataset YAML: {type(names)}")


def _split_list(cfg: dict, split: str) -> List[str]:
    """Return the list of entries (files or directories) for a dataset split.

    YOLO-style YAMLs often define ``train`` / ``val`` / ``test`` either as a
    path (folder or listing file) or as a list of such paths.
    """
    entry = cfg.get(split, None)
    if entry is None:
        raise KeyError(f"Split '{split}' not found in dataset YAML")
    if isinstance(entry, (list, tuple)):
        return [str(e) for e in entry]
    return [str(entry)]


def _iter_image_paths(root: Path, listing: Sequence[str]) -> List[Path]:
    """Expand every split entry into a concrete list of image files.

    Entries may be absolute or relative to ``root`` (the dataset ``path``).
    Entries may point to a directory (recurse for supported extensions) or to
    a listing text file with one image path per line.
    """
    results: List[Path] = []
    for entry in listing:
        p = Path(entry)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            for ext in IMG_EXTS:
                results.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file() and p.suffix.lower() in (".txt",):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    q = Path(line)
                    if not q.is_absolute():
                        q = (root / q).resolve()
                    if q.exists():
                        results.append(q)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            results.append(p)
        else:
            log.warning("Skipping split entry (not a dir/txt/image): %s", p)
    # de-dupe while preserving order
    seen = set()
    uniq: List[Path] = []
    for q in results:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _rgb_to_label(rgb_path: Path) -> Path:
    """Return the YOLO label path for a given image path by replacing
    ``/images/`` with ``/labels/`` (first occurrence from the right) and
    swapping the extension for ``.txt``."""
    parts = list(rgb_path.parts)
    # replace the *last* 'images' directory segment with 'labels'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            break
    else:
        # fall back: ``.../<something>/<image>`` -> ``.../labels/<image>.txt``
        parts = parts[:-2] + ["labels", parts[-1]]
    return Path(*parts).with_suffix(".txt")


def _rgb_to_ir(rgb_path: Path, rules: Sequence[str]) -> Path:
    """Derive the IR counterpart of an RGB path by applying one or more
    ``OLD:NEW`` substitution rules on the string path."""
    s = str(rgb_path)
    for rule in rules:
        if ":" not in rule:
            continue
        old, new = rule.split(":", 1)
        s = s.replace(old, new)
    return Path(s)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

@dataclass
class GTBox:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = int(max(0.0, (self.cx - self.w / 2.0) * img_w))
        y1 = int(max(0.0, (self.cy - self.h / 2.0) * img_h))
        x2 = int(min(float(img_w), (self.cx + self.w / 2.0) * img_w))
        y2 = int(min(float(img_h), (self.cy + self.h / 2.0) * img_h))
        return x1, y1, x2, y2


def _read_labels(label_path: Path, nc: int) -> List[GTBox]:
    if not label_path.exists():
        return []
    boxes: List[GTBox] = []
    with open(label_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if cls < 0 or cls >= nc:
                continue
            if not (0.0 < w <= 1.5 and 0.0 < h <= 1.5):
                continue
            boxes.append(GTBox(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


# ---------------------------------------------------------------------------
# CLIP loading / encoding
# ---------------------------------------------------------------------------

def _load_clip(model_name: str, device: torch.device):
    try:
        import clip  # openai clip
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This script requires the OpenAI CLIP package. Install it via\n"
            "    pip install git+https://github.com/openai/CLIP.git\n"
        ) from exc

    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, clip


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _crop_pil(img: Image.Image, box: GTBox, min_side: int = 8) -> Optional[Image.Image]:
    w, h = img.size
    x1, y1, x2, y2 = box.xyxy(w, h)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return img.crop((x1, y1, x2, y2))


@torch.no_grad()
def _encode_crops(
    model, preprocess, crops: List[Image.Image], device: torch.device, batch_size: int
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack([preprocess(c) for c in crops[i : i + batch_size]]).to(device)
        f = model.encode_image(batch).float()
        feats.append(f.cpu())
    if not feats:
        return torch.empty(0, 512)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _encode_class_names(model, clip_module, names: Sequence[str], device: torch.device) -> torch.Tensor:
    """Encode class names with the CLIP text tower.

    Uses the same ``a photo of a {name}`` prompt convention CLIP is trained
    with, consistent with the description-generation pipeline.
    """
    prompts = [f"a photo of a {n}" for n in names]
    tokens = clip_module.tokenize(prompts).to(device)
    feats = model.encode_text(tokens).float().cpu()
    return feats


# ---------------------------------------------------------------------------
# Prototype accumulation
# ---------------------------------------------------------------------------

@dataclass
class _ProtoAccum:
    nc: int
    dim: int = 512
    sum_: torch.Tensor = field(init=False)
    count: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.sum_ = torch.zeros(self.nc, self.dim)
        self.count = torch.zeros(self.nc, dtype=torch.long)

    def add(self, feats: torch.Tensor, classes: Sequence[int]) -> None:
        if feats.numel() == 0:
            return
        f = F.normalize(feats, dim=-1)  # already 512-d CLIP image features
        for k, c in enumerate(classes):
            self.sum_[c] += f[k]
            self.count[c] += 1

    def protos(self) -> torch.Tensor:
        """Return per-class prototypes, L2-normalised; zero rows if count==0."""
        out = torch.zeros_like(self.sum_)
        nz = self.count > 0
        out[nz] = F.normalize(self.sum_[nz] / self.count[nz].unsqueeze(-1).float(), dim=-1)
        return out


# ---------------------------------------------------------------------------
# Main adaptation
# ---------------------------------------------------------------------------

def _blend(anchor: torch.Tensor, proto: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return ``normalize((1-alpha)*anchor + alpha*proto)`` row-wise.

    Rows for which ``proto`` is the zero vector (i.e. no samples were found
    for that class) keep the anchor unchanged.
    """
    if anchor.shape != proto.shape:
        raise ValueError(f"Shape mismatch: anchor={anchor.shape}, proto={proto.shape}")
    a = F.normalize(anchor, dim=-1)
    p = proto  # already normalised or zero
    has_proto = p.abs().sum(dim=-1) > 0
    blended = a.clone()
    blended[has_proto] = (1.0 - alpha) * a[has_proto] + alpha * p[has_proto]
    return F.normalize(blended, dim=-1)


def _blend_descriptions(
    desc_clip: torch.Tensor,
    class_map: torch.Tensor,
    class_proto: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Per-description blending using the prototype of that description's class."""
    if desc_clip.dim() != 2:
        raise ValueError(f"desc_clip must be (N, C); got {tuple(desc_clip.shape)}")
    if class_map.shape[0] != desc_clip.shape[0]:
        raise ValueError(
            f"class_map ({tuple(class_map.shape)}) does not match desc embeddings "
            f"({tuple(desc_clip.shape)})"
        )
    a = F.normalize(desc_clip, dim=-1)
    out = a.clone()
    for i in range(desc_clip.shape[0]):
        c = int(class_map[i].item())
        if c < 0 or c >= class_proto.shape[0]:
            continue
        proto = class_proto[c]
        if proto.abs().sum() == 0:
            continue
        mixed = (1.0 - alpha) * a[i] + alpha * proto
        out[i] = F.normalize(mixed, dim=0)
    return out


def _load_desc_file(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a per-class description file and return (embeddings, class_map).

    The file is expected to be saved by ``generate_prompts.py`` in its
    per-class mode, i.e. a dict with keys ``embeddings`` (N, 512) and
    ``class_map`` (N,). A plain tensor is also tolerated for backward
    compatibility but requires ``class_map`` to be provided via a companion
    ``.class_map.pt`` file with the same stem; otherwise loading fails.
    """
    data = torch.load(str(path), map_location="cpu")
    if isinstance(data, dict):
        emb = data["embeddings"]
        cmap = data.get("class_map", None)
        if cmap is None:
            raise ValueError(
                f"{path} is a dict but has no 'class_map' key; "
                "per-class mode requires a class_map tensor."
            )
        return emb.float(), cmap.long()
    if torch.is_tensor(data):
        companion = path.with_suffix(".class_map.pt")
        if not companion.exists():
            raise ValueError(
                f"{path} is a plain tensor with no accompanying class_map; "
                "re-run generate_prompts.py with --mode perclass to produce "
                "dict-format embeddings."
            )
        cmap = torch.load(str(companion), map_location="cpu").long()
        return data.float(), cmap
    raise ValueError(f"Unsupported content in {path}: {type(data)}")


def run_adaptation(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset ----------------------------------------------------------
    data_yaml_path = Path(args.data).resolve()
    cfg = _load_yaml(data_yaml_path)
    class_names = _resolve_class_names(cfg)
    nc = len(class_names)
    log.info("Dataset '%s' with %d classes: %s", data_yaml_path.name, nc, class_names)

    root = Path(cfg.get("path", data_yaml_path.parent))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    all_imgs: List[Path] = []
    for sp in splits:
        listing = _split_list(cfg, sp)
        imgs = _iter_image_paths(root, listing)
        log.info("Split '%s': %d images", sp, len(imgs))
        all_imgs.extend(imgs)

    if args.max_images > 0:
        random.shuffle(all_imgs)
        all_imgs = all_imgs[: args.max_images]
        log.info("Using up to %d images (max_images)", len(all_imgs))

    if not all_imgs:
        raise RuntimeError("No images found for the requested split(s).")

    ir_rules = args.ir_replace or ["images:images_ir"]
    log.info("IR substitution rules: %s", ir_rules)

    # ---- CLIP model -------------------------------------------------------
    log.info("Loading CLIP model: %s on %s", args.clip_model, device)
    model, preprocess, clip_module = _load_clip(args.clip_model, device)

    # ---- visual prototypes ------------------------------------------------
    rgb_accum = _ProtoAccum(nc=nc)
    ir_accum = _ProtoAccum(nc=nc)

    per_class_cap = args.max_boxes_per_class if args.max_boxes_per_class > 0 else None

    def _at_cap(accum: _ProtoAccum) -> bool:
        if per_class_cap is None:
            return False
        return bool((accum.count >= per_class_cap).all().item())

    buf_rgb_imgs: List[Image.Image] = []
    buf_rgb_cls: List[int] = []
    buf_ir_imgs: List[Image.Image] = []
    buf_ir_cls: List[int] = []

    def _flush():
        nonlocal buf_rgb_imgs, buf_rgb_cls, buf_ir_imgs, buf_ir_cls
        if buf_rgb_imgs:
            feats = _encode_crops(model, preprocess, buf_rgb_imgs, device, args.batch_size)
            rgb_accum.add(feats, buf_rgb_cls)
            buf_rgb_imgs, buf_rgb_cls = [], []
        if buf_ir_imgs:
            feats = _encode_crops(model, preprocess, buf_ir_imgs, device, args.batch_size)
            ir_accum.add(feats, buf_ir_cls)
            buf_ir_imgs, buf_ir_cls = [], []

    processed = 0
    for rgb_path in all_imgs:
        label_path = _rgb_to_label(rgb_path)
        boxes = _read_labels(label_path, nc=nc)
        if not boxes:
            continue

        ir_path = _rgb_to_ir(rgb_path, ir_rules)
        try:
            rgb_img = _ensure_rgb(Image.open(rgb_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to open RGB image %s: %s", rgb_path, exc)
            continue

        ir_img: Optional[Image.Image] = None
        if ir_path.exists():
            try:
                ir_img = _ensure_rgb(Image.open(ir_path))
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to open IR image %s: %s", ir_path, exc)
                ir_img = None

        for box in boxes:
            if per_class_cap is not None and rgb_accum.count[box.cls].item() >= per_class_cap and \
               (ir_img is None or ir_accum.count[box.cls].item() >= per_class_cap):
                continue
            crop_rgb = _crop_pil(rgb_img, box, min_side=args.min_box_side)
            if crop_rgb is not None:
                if per_class_cap is None or rgb_accum.count[box.cls].item() < per_class_cap:
                    buf_rgb_imgs.append(crop_rgb)
                    buf_rgb_cls.append(box.cls)
            if ir_img is not None:
                crop_ir = _crop_pil(ir_img, box, min_side=args.min_box_side)
                if crop_ir is not None:
                    if per_class_cap is None or ir_accum.count[box.cls].item() < per_class_cap:
                        buf_ir_imgs.append(crop_ir)
                        buf_ir_cls.append(box.cls)

        if len(buf_rgb_imgs) >= args.batch_size or len(buf_ir_imgs) >= args.batch_size:
            _flush()

        processed += 1
        if processed % max(1, args.log_every) == 0:
            log.info(
                "Processed %d images | RGB boxes/class (min,mean,max)=(%d,%.1f,%d) | "
                "IR boxes/class (min,mean,max)=(%d,%.1f,%d)",
                processed,
                int(rgb_accum.count.min()), float(rgb_accum.count.float().mean()), int(rgb_accum.count.max()),
                int(ir_accum.count.min()), float(ir_accum.count.float().mean()), int(ir_accum.count.max()),
            )
        if _at_cap(rgb_accum) and _at_cap(ir_accum):
            log.info("All classes reached max_boxes_per_class cap; stopping early.")
            break

    _flush()

    log.info("Finished crop encoding: RGB total=%d, IR total=%d", int(rgb_accum.count.sum()), int(ir_accum.count.sum()))
    if (rgb_accum.count == 0).any():
        missing = [class_names[i] for i in (rgb_accum.count == 0).nonzero(as_tuple=False).flatten().tolist()]
        log.warning("Classes without any RGB crops (CLIP anchor only): %s", missing)
    if (ir_accum.count == 0).any():
        missing = [class_names[i] for i in (ir_accum.count == 0).nonzero(as_tuple=False).flatten().tolist()]
        log.warning("Classes without any IR crops  (CLIP anchor only): %s", missing)

    class_proto_rgb = rgb_accum.protos()  # (nc, 512)
    class_proto_ir = ir_accum.protos()    # (nc, 512)

    # combined class prototype (fallback to whichever modality has data)
    has_rgb = class_proto_rgb.abs().sum(dim=-1) > 0
    has_ir = class_proto_ir.abs().sum(dim=-1) > 0
    combined = torch.zeros_like(class_proto_rgb)
    both = has_rgb & has_ir
    only_rgb = has_rgb & (~has_ir)
    only_ir = (~has_rgb) & has_ir
    combined[both] = F.normalize(class_proto_rgb[both] + class_proto_ir[both], dim=-1)
    combined[only_rgb] = class_proto_rgb[only_rgb]
    combined[only_ir] = class_proto_ir[only_ir]

    # ---- CLIP class embeddings (anchor) ----------------------------------
    class_clip = _encode_class_names(model, clip_module, class_names, device)  # (nc, 512)

    # ---- desc anchors -----------------------------------------------------
    desc_rgb_path = Path(args.desc_rgb)
    desc_ir_path = Path(args.desc_ir)
    desc_rgb_clip, desc_rgb_cmap = _load_desc_file(desc_rgb_path)
    desc_ir_clip, desc_ir_cmap = _load_desc_file(desc_ir_path)

    # ---- blending ---------------------------------------------------------
    log.info("Blending class embeddings (alpha_class=%.3f)", args.alpha_class)
    class_adapted = _blend(class_clip, combined, args.alpha_class)  # (nc, 512)

    log.info("Blending RGB description embeddings (alpha_desc=%.3f)", args.alpha_desc)
    desc_rgb_adapted = _blend_descriptions(desc_rgb_clip, desc_rgb_cmap, class_proto_rgb, args.alpha_desc)

    log.info("Blending IR description embeddings  (alpha_desc=%.3f)", args.alpha_desc)
    desc_ir_adapted = _blend_descriptions(desc_ir_clip, desc_ir_cmap, class_proto_ir, args.alpha_desc)

    # ---- sanity: delta norms ---------------------------------------------
    def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)

    cos_class = _cos(class_clip, class_adapted)
    cos_desc_rgb = _cos(desc_rgb_clip, desc_rgb_adapted)
    cos_desc_ir = _cos(desc_ir_clip, desc_ir_adapted)
    log.info(
        "Post-adaptation cosine to CLIP anchor | class: mean=%.4f min=%.4f | "
        "desc_rgb: mean=%.4f min=%.4f | desc_ir: mean=%.4f min=%.4f",
        float(cos_class.mean()), float(cos_class.min()),
        float(cos_desc_rgb.mean()), float(cos_desc_rgb.min()),
        float(cos_desc_ir.mean()), float(cos_desc_ir.min()),
    )
    if float(cos_class.min()) < 0.90:
        log.warning(
            "Some class embeddings moved quite far from their CLIP anchor "
            "(min cosine %.3f < 0.90). Consider lowering --alpha-class.",
            float(cos_class.min()),
        )
    if float(cos_desc_rgb.min()) < 0.80 or float(cos_desc_ir.min()) < 0.80:
        log.warning(
            "Some desc embeddings moved quite far from their CLIP anchor; "
            "consider lowering --alpha-desc."
        )

    # ---- save -------------------------------------------------------------
    torch.save(class_adapted.contiguous(), out_dir / "class_adapted_embeddings.pt")
    torch.save(
        {"embeddings": desc_rgb_adapted.contiguous(), "class_map": desc_rgb_cmap.long()},
        out_dir / "desc_rgb_adapted_embeddings.pt",
    )
    torch.save(
        {"embeddings": desc_ir_adapted.contiguous(), "class_map": desc_ir_cmap.long()},
        out_dir / "desc_ir_adapted_embeddings.pt",
    )
    log.info("Saved adapted embeddings to %s", out_dir)

    if args.save_stats:
        stats = {
            "class_names": class_names,
            "alpha_class": args.alpha_class,
            "alpha_desc": args.alpha_desc,
            "rgb_box_count": rgb_accum.count.clone(),
            "ir_box_count": ir_accum.count.clone(),
            "class_proto_rgb": class_proto_rgb,
            "class_proto_ir": class_proto_ir,
            "class_proto_combined": combined,
            "class_clip": class_clip,
            "class_adapted": class_adapted,
            "cos_class_to_clip": cos_class,
            "cos_desc_rgb_to_clip": cos_desc_rgb,
            "cos_desc_ir_to_clip": cos_desc_ir,
        }
        torch.save(stats, out_dir / "prototype_stats.pt")
        log.info("Saved prototype_stats.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline CLIP-anchor + dataset-prototype embedding adapter "
        "for RGBT AR training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Path to dataset YAML file.")
    p.add_argument("--desc-rgb", required=True, help="Path to original RGB description embeddings .pt")
    p.add_argument("--desc-ir", required=True, help="Path to original IR description embeddings .pt")
    p.add_argument("--output-dir", required=True, help="Directory to write adapted embeddings into.")

    p.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64, help="CLIP image encoder batch size.")

    p.add_argument("--alpha-class", type=float, default=0.1,
                   help="Mixing weight for class embeddings (CLIP anchor dominates).")
    p.add_argument("--alpha-desc", type=float, default=0.2,
                   help="Mixing weight for description embeddings.")

    p.add_argument("--split", default="train",
                   help="Which dataset split(s) to scan. Comma-separated.")
    p.add_argument("--ir-replace", action="append", default=None,
                   help="Substitution rule OLD:NEW used to derive IR paths from RGB paths. "
                        "May be repeated. Default: images:images_ir")

    p.add_argument("--max-images", type=int, default=0,
                   help="Upper bound on number of images scanned (0 = no cap).")
    p.add_argument("--max-boxes-per-class", type=int, default=0,
                   help="Upper bound on number of boxes accumulated per class, per modality "
                        "(0 = no cap).")
    p.add_argument("--min-box-side", type=int, default=8,
                   help="Minimum bounding-box side length (pixels) to include in prototype.")

    p.add_argument("--save-stats", action="store_true", default=True,
                   help="Save prototype_stats.pt for debugging.")
    p.add_argument("--no-save-stats", dest="save_stats", action="store_false")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_adaptation(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        sys.exit(130)
