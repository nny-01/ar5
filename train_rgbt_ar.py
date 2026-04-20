# ultralytics/models/yolo/world/train_rgbt_ar.py

import logging
from pathlib import Path
import torch
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RGBTARDetModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.modules.block import AR
logger = logging.getLogger(__name__)


def set_shared_ar_embeddings(model, rel_path=None, desc_rgb_path=None, desc_ir_path=None,
                             desc_rgb_class_map_path=None, desc_ir_class_map_path=None):
    """
    Load shared AR embeddings into model and sync to all AR modules.

    Args:
        model: RGBTARDetModel or de_parallel(model)
        rel_path (str | None): path to shared relational embeddings .pt
        desc_rgb_path (str | None): path to shared RGB descriptive embeddings .pt
        desc_ir_path (str | None): path to shared IR descriptive embeddings .pt
        desc_rgb_class_map_path (str | None): path to RGB desc->class map .pt
        desc_ir_class_map_path (str | None): path to IR desc->class map .pt
    """
    model = de_parallel(model)

    # 1) shared relational embeddings (optional)
    if rel_path:
        rel_path = str(rel_path)
        if not Path(rel_path).exists():
            raise FileNotFoundError(f"[AR] rel_path 不存在: {rel_path}")
        model.load_relational_embeddings(rel_path)
        LOGGER.info(f"[AR] 已加载共享关系嵌入: {rel_path}")

    # 2) shared descriptive embeddings (must be provided together if used)
    if desc_rgb_path or desc_ir_path:
        if not (desc_rgb_path and desc_ir_path):
            raise ValueError(
                "[AR] desc_rgb_path 和 desc_ir_path 必须同时提供，或者同时为 None。"
            )
        desc_rgb_path = str(desc_rgb_path)
        desc_ir_path = str(desc_ir_path)

        if not Path(desc_rgb_path).exists():
            raise FileNotFoundError(f"[AR] desc_rgb_path 不存在: {desc_rgb_path}")
        if not Path(desc_ir_path).exists():
            raise FileNotFoundError(f"[AR] desc_ir_path 不存在: {desc_ir_path}")

        model.load_descriptive_embeddings(desc_rgb_path, desc_ir_path)
        LOGGER.info(
            f"[AR] 已加载共享描述嵌入: RGB={desc_rgb_path}, IR={desc_ir_path}"
        )

    # 3) desc->class maps (separate files, optional)
    if desc_rgb_class_map_path and desc_ir_class_map_path:
        desc_rgb_class_map_path = str(desc_rgb_class_map_path)
        desc_ir_class_map_path = str(desc_ir_class_map_path)
        if not Path(desc_rgb_class_map_path).exists():
            raise FileNotFoundError(f"[AR] desc_rgb_class_map_path 不存在: {desc_rgb_class_map_path}")
        if not Path(desc_ir_class_map_path).exists():
            raise FileNotFoundError(f"[AR] desc_ir_class_map_path 不存在: {desc_ir_class_map_path}")
        if hasattr(model, 'load_desc_class_maps'):
            model.load_desc_class_maps(desc_rgb_class_map_path, desc_ir_class_map_path)
            LOGGER.info(f"[AR] 已加载描述类别映射: RGB={desc_rgb_class_map_path}, IR={desc_ir_class_map_path}")


class RGBTARTrainer(DetectionTrainer):
    """
    DetectionTrainer for RGBT MidFusion + pre-fusion AR.

    Design:
        - Uses standard Detect head, not WorldDetect
        - AR modules consume stored class / relation / description embeddings
        - No per-batch texts / txt_feats injection is required

    Pipeline:
        1. build RGBT paired dataset
        2. build RGBTARDetModel
        3. on_pretrain_routine_end:
           - set class embeddings from dataset names
           - load shared relation / desc embeddings into AR modules
           - sync same info to EMA
        4. preprocess_batch:
           - image normalization only
           - no texts / txt_feats injection
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

        # optional embedding paths; usually mounted externally by user.py
        self.rel_path = getattr(self, "rel_path", None)
        self.desc_rgb_path = getattr(self, "desc_rgb_path", None)
        self.desc_ir_path = getattr(self, "desc_ir_path", None)
        self.desc_rgb_class_map_path = getattr(self, "desc_rgb_class_map_path", None)
        self.desc_ir_class_map_path = getattr(self, "desc_ir_class_map_path", None)
        self.rel_class_map_path = getattr(self, "rel_class_map_path", None)  # compatibility only
        self.add_callback("on_pretrain_routine_end", self._on_pretrain_routine_end_cb)
    # ... 其他方法保持不变（get_model, build_dataset, preprocess_batch 等） ...
    def _on_pretrain_routine_end_cb(self, trainer=None):
        self.on_pretrain_routine_end()
    def get_model(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import RGBTARDetModel
        model = RGBTARDetModel(
            cfg or self.args.model,
            ch=4,                           # RGBT 4通道
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model
    def on_pretrain_routine_end(self):
        model = de_parallel(self.model)

        # 解包 DDP，得到实际模型
        model = de_parallel(self.model)
        print(f"[DEBUG] model type: {type(model).__name__}")
        print(f"[DEBUG] has txt_feats: {hasattr(model, 'txt_feats')}")
        print(f"[DEBUG] has predict override: {'predict' in type(model).__dict__}")
        # 1) 设置类别嵌入
        names = self.data["names"]
        if isinstance(names, dict):
            class_names = [names[k] for k in sorted(names.keys())]
        else:
            class_names = list(names)
        model.set_classes(class_names)
        LOGGER.info(f"[AR] 已设置类别嵌入，共 {len(class_names)} 类")

        # 2) 读取配置路径
        rel_path = getattr(self, "rel_path", None) or getattr(self.args, "rel_path", None)
        desc_rgb_path = getattr(self, "desc_rgb_path", None) or getattr(self.args, "desc_rgb_path", None)
        desc_ir_path = getattr(self, "desc_ir_path", None) or getattr(self.args, "desc_ir_path", None)
        desc_rgb_cmap_path = getattr(self, "desc_rgb_class_map_path", None) or getattr(self.args, "desc_rgb_class_map_path", None)
        desc_ir_cmap_path = getattr(self, "desc_ir_class_map_path", None) or getattr(self.args, "desc_ir_class_map_path", None)

        # 3) 加载关系嵌入（可选）
        if rel_path:
            model.load_relational_embeddings(rel_path, rel_class_map_path=None)
            LOGGER.info(f"[AR] 已加载共享关系嵌入: {rel_path}")

        # 4) 加载描述性嵌入（必须同时提供）
        if desc_rgb_path and desc_ir_path:
            model.load_descriptive_embeddings(desc_rgb_path, desc_ir_path)
            LOGGER.info(f"[AR] 已加载共享描述嵌入: RGB={desc_rgb_path}, IR={desc_ir_path}")
        elif desc_rgb_path or desc_ir_path:
            raise ValueError("[AR] desc_rgb_path 和 desc_ir_path 必须同时提供。")

        # 4.5) 加载描述->类别映射（单独文件，可选）
        if desc_rgb_cmap_path and desc_ir_cmap_path:
            if hasattr(model, 'load_desc_class_maps'):
                model.load_desc_class_maps(desc_rgb_cmap_path, desc_ir_cmap_path)
                LOGGER.info(f"[AR] 已加载描述类别映射: RGB={desc_rgb_cmap_path}, IR={desc_ir_cmap_path}")

        # 5) 同步到 EMA
        if getattr(self, "ema", None) and getattr(self.ema, "ema", None) is not None:
            ema_model = de_parallel(self.ema.ema)
            ema_model.set_classes(class_names)
            if rel_path:
                ema_model.load_relational_embeddings(rel_path, rel_class_map_path=None)
            if desc_rgb_path and desc_ir_path:
                ema_model.load_descriptive_embeddings(desc_rgb_path, desc_ir_path)
            if desc_rgb_cmap_path and desc_ir_cmap_path and hasattr(ema_model, 'load_desc_class_maps'):
                ema_model.load_desc_class_maps(desc_rgb_cmap_path, desc_ir_cmap_path)
            LOGGER.info("[AR] 已同步嵌入到 EMA 模型")

        # 6) 开启 AR 调试（可选）并打印状态
        ar_count = 0
        for m in model.modules():
            if isinstance(m, AR):
                if hasattr(m, "set_debug"):
                    m.set_debug(True, debug_every=20)
                ar_count += 1
                LOGGER.info(f"AR module {ar_count}: class={m.class_embeddings is not None}, "
                            f"rel={m.rel_embeddings is not None}, "
                            f"desc_rgb={m.desc_rgb_embeddings is not None}, "
                            f"desc_ir={m.desc_ir_embeddings is not None}, "
                            f"desc_rgb_cmap={m.desc_rgb_class_map is not None}, "
                            f"desc_ir_cmap={m.desc_ir_class_map is not None}")

        LOGGER.info(f"[AR] 共找到 {ar_count} 个 AR 模块")
    def final_eval(self):
        """Override final_eval: load best weights into in-memory model, re-inject embeddings, validate directly."""
        from ultralytics.utils.torch_utils import strip_optimizer

        # 1) Strip optimizer from saved checkpoints
        for f in (self.last, self.best):
            if f.exists():
                strip_optimizer(f)

        # 2) Load best checkpoint weights into the in-memory model
        if self.best.exists():
            best_ckpt = torch.load(self.best, map_location="cpu")
            best_model = best_ckpt["model"] if isinstance(best_ckpt, dict) and "model" in best_ckpt else best_ckpt
            if hasattr(best_model, "state_dict"):
                state = best_model.float().state_dict()
            else:
                state = best_model
            de_parallel(self.model).load_state_dict(state, strict=False)

            # 3) Re-inject AR embeddings (plain attributes not in state_dict)
            model = de_parallel(self.model)
            names = self.data["names"]
            class_names = [names[k] for k in sorted(names.keys())] if isinstance(names, dict) else list(names)
            model.set_classes(class_names)

            rel_path = getattr(self, "rel_path", None)
            desc_rgb_path = getattr(self, "desc_rgb_path", None)
            desc_ir_path = getattr(self, "desc_ir_path", None)
            desc_rgb_cmap_path = getattr(self, "desc_rgb_class_map_path", None)
            desc_ir_cmap_path = getattr(self, "desc_ir_class_map_path", None)
            if rel_path:
                model.load_relational_embeddings(rel_path)
            if desc_rgb_path and desc_ir_path:
                model.load_descriptive_embeddings(desc_rgb_path, desc_ir_path)
            if desc_rgb_cmap_path and desc_ir_cmap_path and hasattr(model, 'load_desc_class_maps'):
                model.load_desc_class_maps(desc_rgb_cmap_path, desc_ir_cmap_path)

            LOGGER.info("[AR] final_eval: best.pt weights + embeddings loaded into in-memory model")

                # 4) 验证
            self.validator = self.get_validator()
            self.validator.args.plots = True
            self.validator.args.verbose = True
            self.metrics = self.validator(self)
            self.fitness = self.metrics.pop("fitness", 0.0)

            # 5) 强制打印每类别结果（validator 在 training=True 时会跳过）
            self.validator.training = False
            self.validator.print_results()
            
            self.fitness = self.metrics.pop("fitness", -self.loss.detach().cpu().numpy())
            LOGGER.info("[AR] Final evaluation completed")