#!/usr/bin/env python3
"""
MidFusion + AR 训练脚本（适配新版类别补偿 AR）

使用双分支 backbone（RGB + IR 分别提取特征）在 neck 阶段融合，
并在 neck 的第一个 C2f 位置替换为 AR（AlignmentRegion）模块。

当前版本适配"类别补偿 AR"：
  - 所有类别共用一组关系嵌入（REL）
  - RGB/IR 描述嵌入为 per-class（每个类别独立描述）
  - desc_class_map 记录每个描述对应的类别索引
  - AR 使用类别嵌入补偿不确信位置（非描述补偿）

用法：
    直接运行即可开始训练，所有参数在下方配置区域修改：
    python train_midfusion_ar.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path（确保本地模块可导入）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.world.train_rgbt_ar import RGBTARTrainer  # noqa: E402

# ============================================================
#  训练配置（直接在这里修改参数）
# ============================================================

# --- 模型与数据 ---
#MODEL = "/home/cvlab1003/zhangnaiyuan/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/models/v8-RGBT/c2f+ar.yaml"   # 模型配置文件
MODEL = "/home/cvlab1003/zhangnaiyuan/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-midfusion-ar.yaml"
DATA = "/home/cvlab1003/zhangnaiyuan/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/M3FD-rgbt.yaml"                         # 数据集配置文件

# --- 训练超参数 ---
EPOCHS = 300            # 训练轮数
BATCH = 16              # batch size
IMGSZ = 768             # 输入图像尺寸
DEVICE = "0"            # 训练设备 ("0", "0,1", "cpu")
WORKERS = 8             # dataloader workers 数量 
LR0 = 0.0008            # 初始学习率
LRF = 0.01              # 最终学习率 (lr0 * lrf)
WEIGHT_DECAY = 0.0005   # 权重衰减
WARMUP_EPOCHS = 5.0     # warmup 轮数
OPTIMIZER = "AdamW"       # 优化器: "SGD", "Adam", "AdamW"

# --- RGBT 数据加载 ---
USE_SIMOTM = "RGBT"     # 多模态输入模式 (RGBT: RGB+Thermal 4通道)
PAIRS_RGB_IR = True     # 使用 RGB-IR 配对数据加载

# --- 预训练权重 ---
WEIGHTS = "/home/cvlab1003/zhangnaiyuan/YOLOv11-RGBT-master/YOLOv11-RGBT-master/yolov8s.pt"

SKIP_PRETRAINED = False  # 兼容保留；是否生效取决于你的 Trainer 内部实现

# --- AR 描述嵌入（per-class 模式，dict 格式含 class_map）---
# 使用 generate_prompts.py --mode perclass 生成的文件
# 文件格式: {'embeddings': tensor(N, C), 'class_map': tensor(N,)}
# 训练前请先运行 tools/offline_embedding_adapter.py 生成 *_adapted_*.pt，
# 这里再切换到 adapted 路径。原始文件（perclass_desc_*.pt）只作为 adapter 的输入。
DESC_RGB = "prompts/M3FD_prompts/adapted/desc_rgb_adapted_embeddings.pt"
DESC_IR = "prompts/M3FD_prompts/adapted/desc_ir_adapted_embeddings.pt"

# --- AR 类别嵌入（离线适配后的 class embedding，替换 CLIP 即时编码）---
# 由 tools/offline_embedding_adapter.py 生成，shape (nc, 512)，L2 归一化。
# 若为 None 则退回 CLIP set_classes() 原始行为。
CLASS_EMB = "prompts/M3FD_prompts/adapted/class_adapted_embeddings.pt"

# --- AR 关系嵌入（可选，所有类别共享）---
REL = "prompts/M3FD_prompts/shared_relational_embeddings.pt"

# 兼容保留：新 AR 默认不需要 REL_MAP
REL_MAP = None

# --- 输出 ---
PROJECT = "runs/midfusion-ar"
NAME = "train"

# --- 其他 ---
RESUME = False
PATIENCE = 50
SEED = 0
MULTI_SCALE = False

# ============================================================
#  以下为训练逻辑，一般不需要修改
# ============================================================


def _check_optional_file(path_str, name):
    """如果路径非空，则检查文件是否存在。"""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[MidFusion-AR] {name} 文件不存在: {p}")
    return str(p)


def main():
    """MidFusion + AR 训练主函数。"""

    # 先做路径检查
    model_path = _check_optional_file(MODEL, "MODEL")
    data_path = _check_optional_file(DATA, "DATA")
    weights_path = _check_optional_file(WEIGHTS, "WEIGHTS") if WEIGHTS else None
    desc_rgb_path = _check_optional_file(DESC_RGB, "DESC_RGB") if DESC_RGB else None
    desc_ir_path = _check_optional_file(DESC_IR, "DESC_IR") if DESC_IR else None
    rel_path = _check_optional_file(REL, "REL") if REL else None
    rel_map_path = _check_optional_file(REL_MAP, "REL_MAP") if REL_MAP else None
    class_emb_path = _check_optional_file(CLASS_EMB, "CLASS_EMB") if CLASS_EMB else None

    # 构建训练参数
    overrides = {
        "model": model_path,
        "data": data_path,
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "workers": WORKERS,
        "lr0": LR0,
        "lrf": LRF,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "optimizer": OPTIMIZER,
        "project": PROJECT,
        "name": NAME,
        "resume": RESUME,
        "patience": PATIENCE,
        "seed": SEED,
        "multi_scale": MULTI_SCALE,
        "use_simotm": USE_SIMOTM,
        "pairs_rgb_ir": PAIRS_RGB_IR,

    }

    # 如果提供了完整模型权重，加载为 pretrained
    if weights_path:
        overrides["pretrained"] = weights_path

    # 初始化 Trainer
    trainer = RGBTARTrainer(overrides=overrides)
    trainer.desc_rgb_path = desc_rgb_path
    trainer.desc_ir_path = desc_ir_path
    trainer.rel_path = rel_path
    trainer.class_emb_path = class_emb_path

    # ✅ 通过环境变量传递（DDP 多卡模式用）
    if desc_rgb_path:
        os.environ["AR_DESC_RGB_PATH"] = desc_rgb_path
    if desc_ir_path:
        os.environ["AR_DESC_IR_PATH"] = desc_ir_path
    if rel_path:
        os.environ["AR_REL_PATH"] = rel_path
    if class_emb_path:
        os.environ["AR_CLASS_EMB_PATH"] = class_emb_path
    # 给 Trainer 挂载嵌入路径
    # ------------------------------------------------------------
    # 描述嵌入：per-class RGB / IR 描述（含 class_map）
    if desc_rgb_path and desc_ir_path:
        trainer.desc_rgb_path = desc_rgb_path
        trainer.desc_ir_path = desc_ir_path
        print("[MidFusion-AR] 描述嵌入已配置（per-class 模式）:")
        print(f"  RGB: {desc_rgb_path}")
        print(f"  IR:  {desc_ir_path}")
    else:
        trainer.desc_rgb_path = None
        trainer.desc_ir_path = None
        print("[MidFusion-AR] 未提供描述嵌入，AR 模块将以 class-only 模式运行")

    # 关系嵌入：共享 relation
    if rel_path:
        trainer.rel_path = rel_path
        print("[MidFusion-AR] 共享关系嵌入已配置:")
        print(f"  Rel: {rel_path}")
    else:
        trainer.rel_path = None
        print("[MidFusion-AR] 未提供关系嵌入")

    # 兼容旧 Trainer：如果它内部还判断 rel_class_map_path，这里保留字段
    if rel_map_path:
        trainer.rel_class_map_path = rel_map_path
        print("[MidFusion-AR] 兼容配置：REL_MAP 已提供")
        print(f"  Map: {rel_map_path}")
    else:
        trainer.rel_class_map_path = None

    # ------------------------------------------------------------
    # 确定 AR 嵌入模式
    # ------------------------------------------------------------
    if desc_rgb_path and desc_ir_path:
        mode = "Per-class Descriptive + Class Compensation"
        if rel_path:
            mode += " + Shared Relational"
    else:
        mode = "Class-only"

    # 打印训练配置摘要
    print(f"\n{'=' * 60}")
    print("MidFusion + AR 训练配置")
    print(f"{'=' * 60}")
    print(f"  模型:      {model_path}")
    print(f"  数据集:    {data_path}")
    print(f"  Epochs:    {EPOCHS}")
    print(f"  Batch:     {BATCH}")
    print(f"  ImgSz:     {IMGSZ}")
    print(f"  Device:    {DEVICE}")
    print(f"  Workers:   {WORKERS}")
    print(f"  Optimizer: {OPTIMIZER} (lr0={LR0}, lrf={LRF})")
    print(f"  输出:      {PROJECT}/{NAME}")
    print(f"  AR 模式:   {mode}")
    print(f"  WEIGHTS:   {weights_path if weights_path else 'None'}")
    print(f"  DESC_RGB:  {desc_rgb_path if desc_rgb_path else 'None'}")
    print(f"  DESC_IR:   {desc_ir_path if desc_ir_path else 'None'}")
    print(f"  CLASS_EMB: {class_emb_path if class_emb_path else 'None (fallback 到 CLIP set_classes)'}")
    print(f"  REL:       {rel_path if rel_path else 'None'}")
    print(f"  REL_MAP:   {rel_map_path if rel_map_path else 'None (new AR 默认不用)'}")
    print(f"  Note:      DESC .pt files use dict format with embedded class_map")
    print(f"{'=' * 60}\n")
    if desc_rgb_path:
        os.environ["AR_DESC_RGB_PATH"] = desc_rgb_path
    if desc_ir_path:
        os.environ["AR_DESC_IR_PATH"] = desc_ir_path
    if rel_path:
        os.environ["AR_REL_PATH"] = rel_path
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()