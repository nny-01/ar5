"""
Offline prompt generation script for RGBT AR.

Supports two modes:
1. --mode shared (legacy): all classes share one descriptive prompt set
2. --mode perclass (new, default): each class gets its own description samples,
   producing desc_class_map for class-based compensation in the new AR design.

Per-class mode output:
    prompts/<dataset>_prompts/perclass_desc_rgb_embeddings.pt   # dict {'embeddings': (N,C), 'class_map': (N,)}
    prompts/<dataset>_prompts/perclass_desc_ir_embeddings.pt    # dict {'embeddings': (N,C), 'class_map': (N,)}
    prompts/<dataset>_prompts/shared_relational_embeddings.pt   # (Nr, C) tensor (unchanged)

Usage:
    # Per-class mode (recommended for new AR):
    python generate_prompts.py \\
        --api-key "sk-xxx" \\
        --data cfg/datasets/M3FD-rgbt.yaml \\
        --mode perclass \\
        --num-descriptive 5 \\
        --encode

    # Legacy shared mode:
    python generate_prompts.py \\
        --api-key "sk-xxx" \\
        --data cfg/datasets/M3FD-rgbt.yaml \\
        --mode shared \\
        --encode
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path so we can import project modules
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_class_names(data_yaml_path):
    """Load class names from dataset YAML."""
    with open(data_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, dict):
        return list(names.values())
    elif isinstance(names, list):
        return names
    else:
        raise ValueError(f"Unexpected 'names' format in {data_yaml_path}: {type(names)}")


def run_shared_mode(args, generator, class_names, output_path):
    """Legacy shared mode: all classes share one prompt set."""
    dummy_name = "shared"

    relational = generator.generate_relational(
        dummy_name,
        num_prompts=args.num_relational,
        scene_context=args.scene_context,
    )
    desc_rgb = generator.generate_rgb_descriptive(
        dummy_name,
        num_prompts=args.num_descriptive,
    )
    desc_ir = generator.generate_ir_descriptive(
        dummy_name,
        num_prompts=args.num_descriptive,
    )

    all_prompts = {
        "mode": "shared",
        "shared": {
            "class_name": "shared",
            "relational": relational,
            "desc_rgb": desc_rgb,
            "desc_ir": desc_ir,
            "all_prompts": relational + desc_rgb + desc_ir,
        }
    }

    if args.encode:
        from ultralytics.nn.modules.clip_text_encoder import CLIPTextEncoder

        logger.info(f"Encoding shared prompts with CLIP {args.clip_model} on {args.device}...")
        encoder = CLIPTextEncoder(model_name=args.clip_model)
        encoder.load_model(args.device)

        embed_dir = output_path.with_suffix("")
        embed_dir.mkdir(parents=True, exist_ok=True)

        prompt_data = all_prompts["shared"]

        # Encode all prompts
        all_list = prompt_data["all_prompts"]
        if all_list:
            txt_feats = encoder.encode(all_list, device=args.device)
            prompt_data["clip_all_shape"] = list(txt_feats.shape)
            embed_path = embed_dir / "shared_all_embeddings.pt"
            torch.save(txt_feats.cpu(), embed_path)
            logger.info(f"  Shared all embeddings: {txt_feats.shape} -> {embed_path}")

        # Encode shared relational prompts
        rel_list = prompt_data.get("relational", [])
        if rel_list:
            rel_feats = encoder.encode(rel_list, device=args.device)
            prompt_data["clip_relational_shape"] = list(rel_feats.shape)
            rel_path = embed_dir / "shared_relational_embeddings.pt"
            torch.save(rel_feats.cpu(), rel_path)
            logger.info(f"  Shared relational embeddings: {rel_feats.shape} -> {rel_path}")

        # Encode shared RGB descriptive prompts
        desc_rgb_list = prompt_data.get("desc_rgb", [])
        if desc_rgb_list:
            rgb_feats = encoder.encode(desc_rgb_list, device=args.device)
            prompt_data["clip_desc_rgb_shape"] = list(rgb_feats.shape)
            rgb_path = embed_dir / "shared_desc_rgb_embeddings.pt"
            torch.save(rgb_feats.cpu(), rgb_path)
            logger.info(f"  Shared RGB desc embeddings: {rgb_feats.shape} -> {rgb_path}")

        # Encode shared IR descriptive prompts
        desc_ir_list = prompt_data.get("desc_ir", [])
        if desc_ir_list:
            ir_feats = encoder.encode(desc_ir_list, device=args.device)
            prompt_data["clip_desc_ir_shape"] = list(ir_feats.shape)
            ir_path = embed_dir / "shared_desc_ir_embeddings.pt"
            torch.save(ir_feats.cpu(), ir_path)
            logger.info(f"  Shared IR desc embeddings: {ir_feats.shape} -> {ir_path}")

    # Save prompts JSON
    generator.save_prompts(all_prompts, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Shared Prompt Generation Complete!")
    print("=" * 60)
    print(f"\nDataset classes (for reference only): {class_names}")

    prompt_data = all_prompts["shared"]
    print("\nShared relational prompts:")
    for p in prompt_data["relational"]:
        print(f"  - {p}")

    print("\nShared RGB descriptive prompts:")
    for p in prompt_data["desc_rgb"]:
        print(f"  - {p}")

    print("\nShared IR descriptive prompts:")
    for p in prompt_data["desc_ir"]:
        print(f"  - {p}")

    print(f"\nSaved JSON to: {output_path}")

    if args.encode:
        embed_dir = output_path.with_suffix("")
        print(f"CLIP embeddings saved to: {embed_dir}/")
        print("\nUse these files in training:")
        print(f"  rel_path='{embed_dir}/shared_relational_embeddings.pt'")
        print(f"  desc_rgb_path='{embed_dir}/shared_desc_rgb_embeddings.pt'")
        print(f"  desc_ir_path='{embed_dir}/shared_desc_ir_embeddings.pt'")
        print("\nNo desc_class_map needed for shared mode.")


def run_perclass_mode(args, generator, class_names, output_path):
    """Per-class mode: each class gets its own descriptions + class_map."""

    # 1) Generate per-class descriptions
    result = generator.generate_perclass_for_dataset(
        class_names,
        num_descriptive=args.num_descriptive,
    )

    # 2) Also generate shared relational prompts (still shared across classes)
    relational = generator.generate_relational(
        "shared",
        num_prompts=args.num_relational,
        scene_context=args.scene_context,
    )

    all_prompts = {
        "mode": "perclass",
        "class_names": class_names,
        "relational": relational,
        "per_class": result["per_class"],
        "all_desc_rgb": result["all_desc_rgb"],
        "all_desc_ir": result["all_desc_ir"],
        "desc_rgb_class_map": result["desc_rgb_class_map"],
        "desc_ir_class_map": result["desc_ir_class_map"],
    }

    if args.encode:
        from ultralytics.nn.modules.clip_text_encoder import CLIPTextEncoder

        logger.info(f"Encoding per-class prompts with CLIP {args.clip_model} on {args.device}...")
        encoder = CLIPTextEncoder(model_name=args.clip_model)
        encoder.load_model(args.device)

        embed_dir = output_path.with_suffix("")
        embed_dir.mkdir(parents=True, exist_ok=True)

        # Encode shared relational
        if relational:
            rel_feats = encoder.encode(relational, device=args.device)
            rel_path = embed_dir / "shared_relational_embeddings.pt"
            torch.save(rel_feats.cpu(), rel_path)
            logger.info(f"  Relational embeddings: {rel_feats.shape} -> {rel_path}")

        # Encode per-class RGB descriptions -> single tensor + class_map
        all_rgb = result["all_desc_rgb"]
        rgb_class_map = result["desc_rgb_class_map"]
        if all_rgb:
            rgb_feats = encoder.encode(all_rgb, device=args.device)
            rgb_cmap = torch.tensor(rgb_class_map, dtype=torch.long)
            rgb_path = embed_dir / "perclass_desc_rgb_embeddings.pt"
            torch.save({
                "embeddings": rgb_feats.cpu(),
                "class_map": rgb_cmap,
            }, rgb_path)
            logger.info(
                f"  Per-class RGB desc embeddings: {rgb_feats.shape}, "
                f"class_map: {rgb_cmap.shape} -> {rgb_path}"
            )
            all_prompts["clip_desc_rgb_shape"] = list(rgb_feats.shape)

        # Encode per-class IR descriptions -> single tensor + class_map
        all_ir = result["all_desc_ir"]
        ir_class_map = result["desc_ir_class_map"]
        if all_ir:
            ir_feats = encoder.encode(all_ir, device=args.device)
            ir_cmap = torch.tensor(ir_class_map, dtype=torch.long)
            ir_path = embed_dir / "perclass_desc_ir_embeddings.pt"
            torch.save({
                "embeddings": ir_feats.cpu(),
                "class_map": ir_cmap,
            }, ir_path)
            logger.info(
                f"  Per-class IR desc embeddings: {ir_feats.shape}, "
                f"class_map: {ir_cmap.shape} -> {ir_path}"
            )
            all_prompts["clip_desc_ir_shape"] = list(ir_feats.shape)

    # Save prompts JSON
    generator.save_prompts(all_prompts, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Per-Class Prompt Generation Complete!")
    print("=" * 60)
    print(f"\nDataset classes: {class_names}")
    print(f"Descriptions per class: {args.num_descriptive}")
    print(f"Total RGB descriptions: {len(result['all_desc_rgb'])}")
    print(f"Total IR descriptions: {len(result['all_desc_ir'])}")

    print("\nShared relational prompts:")
    for p in relational:
        print(f"  - {p}")

    for name in class_names:
        cls_data = result["per_class"][name]
        print(f"\n[{name}] (class_idx={cls_data['class_idx']}):")
        print(f"  RGB: {cls_data['desc_rgb']}")
        print(f"  IR:  {cls_data['desc_ir']}")

    print(f"\nRGB class_map: {result['desc_rgb_class_map']}")
    print(f"IR class_map:  {result['desc_ir_class_map']}")

    print(f"\nSaved JSON to: {output_path}")

    if args.encode:
        embed_dir = output_path.with_suffix("")
        print(f"CLIP embeddings saved to: {embed_dir}/")
        print("\nUse these files in training (user.py):")
        print(f"  REL       = '{embed_dir}/shared_relational_embeddings.pt'")
        print(f"  DESC_RGB  = '{embed_dir}/perclass_desc_rgb_embeddings.pt'")
        print(f"  DESC_IR   = '{embed_dir}/perclass_desc_ir_embeddings.pt'")
        print("\nThe desc .pt files contain dict format: {'embeddings': tensor, 'class_map': tensor}")
        print("tasks.py will auto-detect and extract class_map from these files.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text prompts for RGBT AR via DeepSeek API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-beb5b9a58c6a44c8a6748167cf4a2f82",
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.deepseek.com",
        help="API base URL (default: https://api.deepseek.com)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name (default: deepseek-chat)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/cvlab1003/zhangnaiyuan/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/M3FD-rgbt.yaml",
        help="Path to dataset YAML file (e.g., cfg/datasets/M3FD-rgbt.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="perclass",
        choices=["shared", "perclass"],
        help="Prompt generation mode: 'shared' (legacy) or 'perclass' (new, default)"
    )
    parser.add_argument(
        "--num-relational",
        type=int,
        default=5,
        help="Number of shared relational prompts (default: 5)"
    )
    parser.add_argument(
        "--num-descriptive",
        type=int,
        default=5,
        help="Number of descriptive prompts per class (perclass) or total (shared) (default: 5)"
    )
    parser.add_argument(
        "--scene-context",
        type=str,
        default="RGBT surveillance",
        help="Scene context for shared relational prompts (default: 'RGBT surveillance')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: prompts/<dataset_name>_prompts.json)"
    )
    parser.add_argument(
        "--encode",
        action="store_true",
        default=True,
        help="Also encode prompts with CLIPTextEncoder and save embeddings"
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="CLIP model variant (default: ViT-B/32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for CLIP encoding (default: auto)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature (default: 0.7)"
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("API key required. Use --api-key or set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    # Load class names
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset YAML not found: {data_path}")
        sys.exit(1)

    class_names = load_class_names(str(data_path))
    logger.info(f"Loaded {len(class_names)} classes from {data_path}: {class_names}")
    logger.info(f"Generation mode: {args.mode}")

    # Import generator
    from ultralytics.nn.modules.llm_text_generator import LLMTextGenerator

    generator = LLMTextGenerator(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        dataset_name = data_path.stem.replace("-rgbt", "").replace("-RGBT", "")
        output_path = Path("prompts") / f"{dataset_name}_prompts.json"

    # Run selected mode
    if args.mode == "shared":
        run_shared_mode(args, generator, class_names, output_path)
    else:
        run_perclass_mode(args, generator, class_names, output_path)


if __name__ == "__main__":
    main()