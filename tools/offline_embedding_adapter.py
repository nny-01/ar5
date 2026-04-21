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
Parameters live in the ``CONFIG`` section at the top of this file
(``DATA``, ``DESC_RGB``, ``DESC_IR``, ``OUTPUT_DIR``, ``ALPHA_CLASS``,
``ALPHA_DESC``, ``IR_REPLACE`` ...). Edit those and then just run::

    python offline_embedding_adapter.py

Every CLI flag is optional and, if given, overrides the corresponding
default from the CONFIG section. For example::

    python offline_embedding_adapter.py --alpha-class 0.05 --split train,val
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
from PIL import Image, ImageFile

# Some training datasets contain slightly truncated PNG/JPEG files. PIL
# raises ``OSError: image file is truncated`` on those by default, which
# would abort a full-dataset scan. We tell PIL to decode whatever bytes it
# has and we defensively re-try / skip any image that still fails below.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("offline_embedding_adapter")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ============================================================
# 训练前需要修改的参数 —— 直接改这里即可，无需命令行
# 命令行参数仍然可用，如果传了会覆盖这里的默认值
# ============================================================

# --- 数据集与输入输出路径 ---
DATA = "ultralytics/cfg/datasets/M3FD-rgbt.yaml"                   # 数据集 YAML
# DESC_RGB / DESC_IR 可以是：
#   (a) 一个 .pt 文件（dict 带 class_map，或 tensor+同名 .class_map.pt），
#   (b) 一个目录，里面按类放 {class_name}_desc_{rgb|ir}_embeddings.pt。
# 默认用 (b)，匹配 generate_prompts.py 旧版的 per-class 输出布局。
DESC_RGB = "prompts/M3FD_prompts"                                  # 原始 RGB 描述嵌入
DESC_IR = "prompts/M3FD_prompts"                                   # 原始 IR 描述嵌入
OUTPUT_DIR = "prompts/M3FD_prompts/adapted"                        # adapted 嵌入输出目录

# --- 适配超参数 ---
ALPHA_CLASS = 0.1       # class embedding 融合权重（CLIP 锚点主导，建议 0.1 左右）
# desc embedding 融合权重，RGB 和 IR 可独立设置（建议 0.15 ~ 0.25）。
# 如果你只想用同一个值，把两者设成一样即可。
ALPHA_DESC_RGB = 0.2    # RGB desc 融合权重
ALPHA_DESC_IR = 0.2     # IR  desc 融合权重
ALPHA_DESC = None       # 向后兼容：若非 None，会同时覆盖上面两个。留 None 即可。

# --- 数据扫描与 CLIP ---
SPLIT = "train"                        # 用哪个 split（train / val，可逗号分隔）
IR_REPLACE = ["/vi/:/ir/"]             # 从 RGB 路径推导 IR 路径的替换规则，可给多条
                                       # M3FD_YOLOWORLD 用 /vi/:/ir/；如果你的 IR 放在 images_ir/
                                       # 下请改成 "images:images_ir"；可以叠加多条规则
CLIP_MODEL = "ViT-B/32"                # CLIP 模型名
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
BATCH_SIZE = 64                        # CLIP 图像编码 batch size

# --- 规模控制（调试时可以限一下）---
MAX_IMAGES = 0                         # 扫描图像数上限（0 = 不限制）
MAX_BOXES_PER_CLASS = 0                # 每类每模态 box 数上限（0 = 不限制）
MIN_BOX_SIDE = 8                       # 小于此像素的 box 会被丢弃

# --- 其他 ---
SAVE_STATS = True                      # 保存 prototype_stats.pt
SEED = 0
LOG_EVERY = 200

# ============================================================
# 以下为实现细节，通常不需要修改
# ============================================================


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


def _safe_open_image(path: Path) -> Optional[Image.Image]:
    """Open an image, force-load its pixel data, and convert to RGB.

    Returns ``None`` (and logs a warning) if the file is missing, unreadable,
    or too corrupted to decode even with ``LOAD_TRUNCATED_IMAGES = True``.
    """
    try:
        img = Image.open(path)
        img.load()  # force decoding now so later .crop() cannot raise
        return _ensure_rgb(img)
    except (OSError, ValueError, SyntaxError) as exc:
        log.warning("Skipping unreadable image %s: %s", path, exc)
        return None


def _crop_pil(img: Image.Image, box: GTBox, min_side: int = 8) -> Optional[Image.Image]:
    w, h = img.size
    x1, y1, x2, y2 = box.xyxy(w, h)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    try:
        return img.crop((x1, y1, x2, y2))
    except (OSError, ValueError) as exc:
        log.warning("Skipping corrupt crop in image (box=%s): %s", box, exc)
        return None


@torch.no_grad()
def _encode_crops(
    model, preprocess, crops: List[Image.Image], device: torch.device, batch_size: int
) -> torch.Tensor:
    """Encode a list of PIL crops with the CLIP image encoder.

    Crops that raise during ``preprocess`` (corrupted pixel data) are silently
    dropped rather than aborting the batch; a matching outer list of class
    IDs must be filtered in sync — see ``_encode_crops_aligned``.
    """
    feats: List[torch.Tensor] = []
    for i in range(0, len(crops), batch_size):
        tensors: List[torch.Tensor] = []
        for c in crops[i : i + batch_size]:
            try:
                tensors.append(preprocess(c))
            except (OSError, ValueError) as exc:  # pragma: no cover - defensive
                log.warning("Skipping crop that failed CLIP preprocess: %s", exc)
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        f = model.encode_image(batch).float()
        feats.append(f.cpu())
    if not feats:
        return torch.empty(0, 512)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _encode_crops_aligned(
    model, preprocess, crops: List[Image.Image], cls_ids: List[int],
    device: torch.device, batch_size: int,
) -> Tuple[torch.Tensor, List[int]]:
    """Like ``_encode_crops`` but also returns the filtered class-id list so
    the accumulator stays aligned even if some crops are dropped mid-batch.
    """
    feats: List[torch.Tensor] = []
    kept_cls: List[int] = []
    for i in range(0, len(crops), batch_size):
        tensors: List[torch.Tensor] = []
        batch_cls: List[int] = []
        for c, k in zip(crops[i : i + batch_size], cls_ids[i : i + batch_size]):
            try:
                tensors.append(preprocess(c))
                batch_cls.append(k)
            except (OSError, ValueError) as exc:  # pragma: no cover - defensive
                log.warning("Skipping crop that failed CLIP preprocess: %s", exc)
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        f = model.encode_image(batch).float()
        feats.append(f.cpu())
        kept_cls.extend(batch_cls)
    if not feats:
        return torch.empty(0, 512), []
    return torch.cat(feats, dim=0), kept_cls


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


def _load_desc_file(
    path: Path,
    class_names: Optional[Sequence[str]] = None,
    modality: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load description embeddings and return ``(embeddings, class_map)``.

    Accepts three layouts:

    1. ``path`` is a single ``.pt`` saved as a dict ``{'embeddings', 'class_map'}``
       (the canonical output of ``generate_prompts.py --mode perclass``).
    2. ``path`` is a single ``.pt`` tensor plus a companion file
       ``<stem>.class_map.pt`` at the same location.
    3. ``path`` is a **directory** containing one file per class named
       ``{class_name}_desc_{rgb|ir}_embeddings.pt`` (older ``generate_prompts.py``
       style). The files are concatenated in the order of ``class_names`` and
       a class_map is synthesised from the class index.

    In mode (3) both ``class_names`` and ``modality`` (``"rgb"`` or ``"ir"``)
    must be passed in.
    """
    if path.is_dir():
        if class_names is None or modality not in ("rgb", "ir"):
            raise ValueError(
                f"{path} is a directory but class_names / modality were not "
                "provided to _load_desc_file."
            )
        embs: List[torch.Tensor] = []
        cmap: List[int] = []
        missing: List[str] = []
        for cls_idx, cls_name in enumerate(class_names):
            candidate = path / f"{cls_name}_desc_{modality}_embeddings.pt"
            if not candidate.exists():
                missing.append(cls_name)
                continue
            blob = torch.load(str(candidate), map_location="cpu")
            if isinstance(blob, dict):
                t = blob.get("embeddings", None)
                if t is None:
                    raise ValueError(
                        f"{candidate} is a dict but has no 'embeddings' key"
                    )
            else:
                t = blob
            if not torch.is_tensor(t):
                raise ValueError(
                    f"{candidate} has unsupported content: {type(blob)}"
                )
            t = t.float().reshape(-1, t.shape[-1])
            embs.append(t)
            cmap.extend([cls_idx] * t.shape[0])
            log.info(
                "  loaded %s: %d descriptions (class_id=%d)",
                candidate.name, t.shape[0], cls_idx,
            )
        if not embs:
            raise FileNotFoundError(
                f"No per-class description files found under {path} with "
                f"pattern '{{class}}_desc_{modality}_embeddings.pt'"
            )
        if missing:
            log.warning(
                "Missing per-class description files for %s in %s (modality=%s); "
                "those classes will keep their CLIP anchor without desc blending.",
                missing, path, modality,
            )
        return torch.cat(embs, dim=0), torch.tensor(cmap, dtype=torch.long)

    data = torch.load(str(path), map_location="cpu")
    if isinstance(data, dict):
        emb = data["embeddings"]
        cmap_t = data.get("class_map", None)
        if cmap_t is None:
            raise ValueError(
                f"{path} is a dict but has no 'class_map' key; "
                "per-class mode requires a class_map tensor."
            )
        return emb.float(), cmap_t.long()
    if torch.is_tensor(data):
        companion = path.with_suffix(".class_map.pt")
        if not companion.exists():
            raise ValueError(
                f"{path} is a plain tensor with no accompanying class_map. "
                f"Either (a) re-run generate_prompts.py with --mode perclass to "
                f"produce dict-format embeddings, or (b) point DESC_RGB / DESC_IR "
                f"at the directory that contains per-class files "
                f"'{{class_name}}_desc_<rgb|ir>_embeddings.pt' so the adapter can "
                f"concatenate them itself."
            )
        cmap_t = torch.load(str(companion), map_location="cpu").long()
        return data.float(), cmap_t
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
            feats, kept_cls = _encode_crops_aligned(
                model, preprocess, buf_rgb_imgs, buf_rgb_cls, device, args.batch_size
            )
            if feats.numel():
                rgb_accum.add(feats, kept_cls)
            buf_rgb_imgs, buf_rgb_cls = [], []
        if buf_ir_imgs:
            feats, kept_cls = _encode_crops_aligned(
                model, preprocess, buf_ir_imgs, buf_ir_cls, device, args.batch_size
            )
            if feats.numel():
                ir_accum.add(feats, kept_cls)
            buf_ir_imgs, buf_ir_cls = [], []

    ir_rule_noop_warned = False
    ir_missing_first_examples: List[Tuple[Path, Path]] = []
    processed = 0
    for rgb_path in all_imgs:
        label_path = _rgb_to_label(rgb_path)
        boxes = _read_labels(label_path, nc=nc)
        if not boxes:
            continue

        ir_path = _rgb_to_ir(rgb_path, ir_rules)
        # Surface obvious mis-configurations early so the user doesn't
        # silently end up with zero IR samples.
        if ir_path == rgb_path and not ir_rule_noop_warned:
            log.warning(
                "IR substitution rule(s) %s did not change the RGB path (%s); "
                "IR prototypes will stay empty. Fix IR_REPLACE at the top of the "
                "script (e.g. 'vi:ir' or '/vi/:/ir/') or pass --ir-replace.",
                ir_rules, rgb_path,
            )
            ir_rule_noop_warned = True
        if not ir_path.exists() and len(ir_missing_first_examples) < 3:
            ir_missing_first_examples.append((rgb_path, ir_path))

        rgb_img = _safe_open_image(rgb_path)
        if rgb_img is None:
            continue

        ir_img: Optional[Image.Image] = None
        if ir_path.exists():
            ir_img = _safe_open_image(ir_path)

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
    if int(ir_accum.count.sum()) == 0 and ir_missing_first_examples:
        examples = "\n    ".join(
            f"{r}\n        -> tried IR: {i}" for r, i in ir_missing_first_examples
        )
        log.error(
            "No IR images were found. The current IR substitution rule(s) %s "
            "do not resolve to existing files. Examples:\n    %s\n"
            "Fix IR_REPLACE at the top of the script (e.g. 'vi:ir' for the "
            "M3FD_YOLOWORLD layout) or pass --ir-replace.",
            ir_rules, examples,
        )
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
    log.info("Loading RGB description embeddings from %s", desc_rgb_path)
    desc_rgb_clip, desc_rgb_cmap = _load_desc_file(desc_rgb_path, class_names, "rgb")
    log.info("Loading IR description embeddings from %s", desc_ir_path)
    desc_ir_clip, desc_ir_cmap = _load_desc_file(desc_ir_path, class_names, "ir")

    # ---- blending ---------------------------------------------------------
    log.info("Blending class embeddings (alpha_class=%.3f)", args.alpha_class)
    class_adapted = _blend(class_clip, combined, args.alpha_class)  # (nc, 512)

    log.info("Blending RGB description embeddings (alpha_desc_rgb=%.3f)", args.alpha_desc_rgb)
    desc_rgb_adapted = _blend_descriptions(desc_rgb_clip, desc_rgb_cmap, class_proto_rgb, args.alpha_desc_rgb)

    log.info("Blending IR description embeddings  (alpha_desc_ir=%.3f)", args.alpha_desc_ir)
    desc_ir_adapted = _blend_descriptions(desc_ir_clip, desc_ir_cmap, class_proto_ir, args.alpha_desc_ir)

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
    if float(cos_desc_rgb.min()) < 0.80:
        log.warning(
            "RGB desc embeddings moved quite far from their CLIP anchor "
            "(min cosine %.3f < 0.80). Consider lowering --alpha-desc-rgb.",
            float(cos_desc_rgb.min()),
        )
    if float(cos_desc_ir.min()) < 0.80:
        log.warning(
            "IR desc embeddings moved quite far from their CLIP anchor "
            "(min cosine %.3f < 0.80). Consider lowering --alpha-desc-ir.",
            float(cos_desc_ir.min()),
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
            "alpha_desc_rgb": args.alpha_desc_rgb,
            "alpha_desc_ir": args.alpha_desc_ir,
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
    """Build the CLI parser. All arguments default to the module-level
    constants at the top of this file, so the script can also be run with
    no arguments at all: ``python offline_embedding_adapter.py``.
    """
    p = argparse.ArgumentParser(
        description="Offline CLIP-anchor + dataset-prototype embedding adapter "
        "for RGBT AR training. Defaults come from the CONFIG section at the "
        "top of this file; CLI flags override those defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=DATA, help="Path to dataset YAML file.")
    p.add_argument("--desc-rgb", default=DESC_RGB, help="Path to original RGB description embeddings .pt")
    p.add_argument("--desc-ir", default=DESC_IR, help="Path to original IR description embeddings .pt")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to write adapted embeddings into.")

    p.add_argument("--clip-model", default=CLIP_MODEL, help="CLIP model name.")
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="CLIP image encoder batch size.")

    p.add_argument("--alpha-class", type=float, default=ALPHA_CLASS,
                   help="Mixing weight for class embeddings (CLIP anchor dominates).")
    # Back-compat: --alpha-desc sets both; --alpha-desc-rgb / --alpha-desc-ir override.
    default_desc_rgb = ALPHA_DESC if ALPHA_DESC is not None else ALPHA_DESC_RGB
    default_desc_ir = ALPHA_DESC if ALPHA_DESC is not None else ALPHA_DESC_IR
    p.add_argument("--alpha-desc", type=float, default=None,
                   help="Back-compat: shorthand that sets both --alpha-desc-rgb "
                        "and --alpha-desc-ir to the same value.")
    p.add_argument("--alpha-desc-rgb", type=float, default=default_desc_rgb,
                   help="Mixing weight for RGB description embeddings.")
    p.add_argument("--alpha-desc-ir", type=float, default=default_desc_ir,
                   help="Mixing weight for IR description embeddings.")

    p.add_argument("--split", default=SPLIT,
                   help="Which dataset split(s) to scan. Comma-separated.")
    p.add_argument("--ir-replace", action="append", default=None,
                   help="Substitution rule OLD:NEW used to derive IR paths from RGB paths. "
                        "May be repeated. If not given, falls back to IR_REPLACE at top of file.")

    p.add_argument("--max-images", type=int, default=MAX_IMAGES,
                   help="Upper bound on number of images scanned (0 = no cap).")
    p.add_argument("--max-boxes-per-class", type=int, default=MAX_BOXES_PER_CLASS,
                   help="Upper bound on number of boxes accumulated per class, per modality "
                        "(0 = no cap).")
    p.add_argument("--min-box-side", type=int, default=MIN_BOX_SIDE,
                   help="Minimum bounding-box side length (pixels) to include in prototype.")

    p.add_argument("--save-stats", action="store_true", default=SAVE_STATS,
                   help="Save prototype_stats.pt for debugging.")
    p.add_argument("--no-save-stats", dest="save_stats", action="store_false")

    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    return p


def _validate_config(args: argparse.Namespace) -> None:
    """Raise a helpful error if any required path is still empty.

    This guards the 'zero-argument' entry point where the user is expected
    to have filled in the CONFIG section at the top of the file.
    """
    missing = [
        name for name, value in (
            ("DATA / --data", args.data),
            ("DESC_RGB / --desc-rgb", args.desc_rgb),
            ("DESC_IR / --desc-ir", args.desc_ir),
            ("OUTPUT_DIR / --output-dir", args.output_dir),
        ) if not value
    ]
    if missing:
        raise SystemExit(
            "The following paths are empty — set them in the CONFIG section at the "
            "top of offline_embedding_adapter.py (or pass them on the command line):\n  - "
            + "\n  - ".join(missing)
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    # Fall back to the module-level IR_REPLACE constant if nothing was passed on the CLI.
    if args.ir_replace is None:
        args.ir_replace = list(IR_REPLACE) if IR_REPLACE else None
    # Back-compat: --alpha-desc (or module-level ALPHA_DESC) overrides both modalities.
    if args.alpha_desc is not None:
        args.alpha_desc_rgb = args.alpha_desc
        args.alpha_desc_ir = args.alpha_desc
    elif ALPHA_DESC is not None:
        args.alpha_desc_rgb = ALPHA_DESC
        args.alpha_desc_ir = ALPHA_DESC
    _validate_config(args)
    run_adaptation(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        sys.exit(130)
