# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import thop
import torch
from timm.models.focalnet import FocalModulation


from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1, ELAN, ELAN_H, MP_1, MP_2, ELAN_t, SPPCSPCSIM, SPPELAN, SPPCSPC,
    OBB,
    PSA, CrossAttentionShared, CrossMLCA, TensorSelector, CrossMLCAv2, DeepDiverseBranchBlock, CBAM,
    RecursionDiverseBranchBlock,MANet, HyperComputeModule, MANet_FasterBlock, MANet_FasterCGLU, MANet_Star,
    C3k2_DeepDBB, C3k2_DBB, C3k2_WDBB, C2f_DeepDBB, C2f_WDBB, C2f_DBB, C3k_RDBB, C2f_RDBB, C3k2_RDBB, A2C2f,
    CrossC2f, CrossC3k2,
    GPT,Add2,CrossTransformerFusion,NiNfusion,TransformerFusionBlock,

    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP, YOLOv4_BottleneckCSP, YOLOv4_Bottleneck,
    CBH, ES_Bottleneck, DWConvblock, ADD,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index, Silence, SilenceChannel, ChannelToNumber, NumberToChannel, DiverseBranchBlock, WideDiverseBranchBlock,
    DeepDiverseBranchBlock, FeaturePyramidAggregationAttention, SilenceLayer,ZeroConv2d,ZeroConv1d,
    ConvNormLayer, BasicBlock, BottleNeck, Blocks,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    v10Detect, DetectDeepDBB, DetectWDBB, DetectV8, DetectAux,
    Detect_LSCD, Segment_LSCD, Pose_LSCD, OBB_LSCD,
)

from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

from ultralytics.nn.modules.block import AR, AlignmentRegion
from ultralytics.nn.modules.attention import *
from ultralytics.nn.modules.ppyolo import (
    CSPResNet_CBS, CSPResNet, ConvBNLayer, ResSPP, CoordConv,
    ResNet50vd, ResNet50vd_dcn, ResNet101vd, PPConvBlock, Res2net50,

)
from ultralytics.nn.modules.yolov13_block import (
    DSConv, DSC3k2, DownsampleConv, FullPAD_Tunnel, HyperACE
)

# 代码格式参考 B站 魔鬼面具
DETECT_CLASS = (Detect,  Detect_LSCD, DetectAux, DetectDeepDBB, DetectWDBB,DetectV8)
V10_DETECT_CLASS = (v10Detect,)
SEGMENT_CLASS = (Segment, )
POSE_CLASS = (Pose, Pose_LSCD,)
OBB_CLASS = (OBB, OBB_LSCD,)
C3K2_CLASS = (C3k2)




class BaseModel(torch.nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Run forward pass through all layers sequentially.

        Note: For WorldModel/WorldRGBTModel, this method is NOT called during
        normal training or inference — WorldModel.predict() overrides
        BaseModel.predict() and handles text feature routing (C2fAttn,
        ImagePoolingAttn, WorldDetect) with proper ori_txt_feats separation.

        This method only needs to handle standard (non-World) modules.
        World module handling is kept here as a fallback for edge cases
        (e.g., profiling, embedding extraction) but follows the same
        text feature flow as WorldModel.predict().
        """
        from ultralytics.nn.modules.block import C2fAttn, ImagePoolingAttn, AlignmentRegion
        from ultralytics.nn.modules.head import WorldDetect

        txt_feats = getattr(self, "txt_feats", None)
        ori_txt_feats = txt_feats.clone() if txt_feats is not None else None

        y, dt, embeddings = [], [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)

            if isinstance(m, C2fAttn) and txt_feats is not None:
                x = m(x, txt_feats)
            elif isinstance(m, AlignmentRegion) and txt_feats is not None:
                x = m(x, txt_feats)
            elif isinstance(m, ImagePoolingAttn) and txt_feats is not None:
                # ImagePoolingAttn returns enriched TEXT features, not visual features.
                # Must assign to txt_feats, not x. x (visual feature list) stays unchanged.
                txt_feats = m(x, txt_feats)
            elif isinstance(m, WorldDetect) and ori_txt_feats is not None:
                # WorldDetect uses ORIGINAL (un-modified) text features for contrastive alignment,
                # not the ImagePoolingAttn-enriched version.
                x = m(x, ori_txt_feats)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, ConvNormLayer):
                    m.conv = fuse_conv_and_bn(m.conv, m.norm)  # update conv
                    delattr(m, 'norm')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, DETECT_CLASS):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLO detection model."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLO detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS + V10_DETECT_CLASS)):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            # s = 256  # 2x min stride
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if isinstance(m, (DetectAux,)):
                    return self.forward(x)[:3]
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, SEGMENT_CLASS + POSE_CLASS + OBB_CLASS) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO segmentation model."""

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO pose model."""

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO classification model."""

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn ):
                x = m(x, txt_feats)
            elif isinstance(m, AR):
                desc_rgb = getattr(self, 'desc_rgb_feats', None)
                desc_ir = getattr(self, 'desc_ir_feats', None)
                if desc_rgb is not None:
                    desc_rgb = desc_rgb.to(device=x.device, dtype=x.dtype)
                    if len(desc_rgb) != len(x):
                        desc_rgb = desc_rgb.expand(len(x), -1, -1)
                if desc_ir is not None:
                    desc_ir = desc_ir.to(device=x.device, dtype=x.dtype)
                    if len(desc_ir) != len(x):
                        desc_ir = desc_ir.expand(len(x), -1, -1)
                x = m(x, txt_feats, desc_rgb, desc_ir)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class WorldRGBTModel(WorldModel):
    """YOLOv8 World Model for RGBT (RGB + Thermal) detection with CLIP text guidance.

    Combines EarlyFusion backbone (4-channel RGBT input) with YOLO-World style
    text-guided detection head. Supports both C2fAttn and AR (Alignment Region) modules.

    When using AR module, supports dual-branch projection with:
    - Class embeddings (from CLIP-encoded class names)
    - RGB descriptive embeddings (from LLM-generated RGB appearance prompts)
    - IR descriptive embeddings (from LLM-generated thermal appearance prompts)
    - Cross-modal mapping between RGB and IR descriptive embeddings

    Text feature flow (AR mode):
        CLIPTextEncoder → class_embeds (txt_feats)
        CLIPTextEncoder → desc_rgb_feats (RGB descriptive embeddings)
        CLIPTextEncoder → desc_ir_feats (IR descriptive embeddings)
            ↓
        AR(visual_feats, class_embeds, desc_rgb, desc_ir)
          — dual projection, entropy, threshold routing, cross-modal mapping
            ↓
        ImagePoolingAttn(visual_feats, txt_feats) → enriched txt_feats
            ↓
        WorldDetect(visual_feats, txt_feats) — contrastive classification
    """

    def __init__(self, cfg="yolov8-RGBT-earlyfusion-clip.yaml", ch=4, nc=None, verbose=True):
        """Initialize WorldRGBTModel with 4-channel (RGBT) input and CLIPTextEncoder.

        Args:
            cfg (str | dict): Model config file path or dict.
            ch (int): Input channels (4 for RGBT). Default: 4.
            nc (int | None): Number of classes. Default: None (uses config value).
            verbose (bool): Print model info. Default: True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        from ultralytics.nn.modules.clip_text_encoder import CLIPTextEncoder
        self.clip_text_encoder = CLIPTextEncoder(model_name="ViT-B/32", embed_dim=512)
        # Descriptive embeddings for AR module (loaded via load_descriptive_embeddings)
        self.desc_rgb_feats = None
        self.desc_ir_feats = None

    def _sync_desc_to_ar(self):
        """Push stored descriptive embeddings to all AR modules.

        Ensures every AR module has its own copy of the embeddings via
        register_buffer, so they survive model.to(device) and DDP wrapping.
        """
        print("Push stored descriptive embeddings to all AR modules")
        for m in self.model.modules():
            if isinstance(m, AR):
                m.set_desc_embeddings(self.desc_rgb_feats, self.desc_ir_feats)

    def _sync_class_to_ar(self):
        """Push stored class embeddings to all AR modules.

        Ensures every AR module has its own copy of the class embeddings via
        register_buffer, so they survive model.to(device) and DDP wrapping.
        """
        for m in self.model.modules():
            if isinstance(m, AR):
                m.set_class_embeddings(self.txt_feats)

    def load_descriptive_embeddings(self, rgb_path, ir_path):
        """Load pre-computed CLIP-encoded descriptive embeddings for AR module.

        These embeddings are generated offline by generate_prompts.py and encoded
        by CLIPTextEncoder. They describe how objects appear in RGB and IR modalities.

        Loads into model-level attributes AND pushes to all AR modules directly.

        Args:
            rgb_path (str): Path to RGB descriptive embeddings .pt file.
                Shape: (nd, gc) where nd = number of descriptive prompts, gc = 512.
            ir_path (str): Path to IR descriptive embeddings .pt file.
                Shape: (nd, gc) where nd = number of descriptive prompts, gc = 512.
        """
        import torch
        self.desc_rgb_feats = torch.load(rgb_path, map_location="cpu")
        self.desc_ir_feats = torch.load(ir_path, map_location="cpu")
        # Ensure batch dimension: (nd, gc) → (1, nd, gc)
        if self.desc_rgb_feats.dim() == 2:
            self.desc_rgb_feats = self.desc_rgb_feats.unsqueeze(0)
        if self.desc_ir_feats.dim() == 2:
            self.desc_ir_feats = self.desc_ir_feats.unsqueeze(0)
        # Push to AR modules directly
        self._sync_desc_to_ar()
        LOGGER.info(
            f"Loaded descriptive embeddings into model and {sum(1 for m in self.model.modules() if isinstance(m, AR))} AR module(s): "
            f"RGB={self.desc_rgb_feats.shape}, IR={self.desc_ir_feats.shape}"
        )

    def load_relational_embeddings(self, rel_path, rel_class_map_path):
        """Load pre-computed relational embeddings for AR modules.

        Args:
            rel_path (str): Path to combined relational embeddings .pt file.
            rel_class_map_path (str): Path to relational class mapping .pt file.
        """
        import torch
        rel_embeds = torch.load(rel_path, map_location="cpu")
        rel_class_map = torch.load(rel_class_map_path, map_location="cpu")
        if rel_embeds.dim() == 2:
            rel_embeds = rel_embeds.unsqueeze(0)
        count = 0
        for m in self.model.modules():
            if isinstance(m, AR):
                m.set_relational_embeddings(rel_embeds, rel_class_map)
                count += 1
        LOGGER.info(f"Loaded relational embeddings into {count} AR module(s): "
                     f"rel={rel_embeds.shape}, class_map={rel_class_map.shape}")

    def get_ar_mapping_loss(self):
        """Collect cross-modal mapping losses from all AR modules.

        Uses model.modules() to recursively find AR instances, including those
        nested inside nn.Sequential wrappers (created by parse_model when n > 1).

        Returns:
            torch.Tensor: Sum of mapping losses from all AR modules in the model.
        """
        import torch
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for m in self.model.modules():
            if isinstance(m, AR) and m.mapping_loss is not None:
                total = total + m.mapping_loss
        return total

    def loss(self, batch, preds=None):
        """Compute loss with AR cross-modal mapping loss added.

        Extends WorldModel.loss() by adding the bidirectional mapping loss
        (RGB desc ↔ IR desc) from all AR modules, weighted by mapping_loss_weight.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])

        loss, loss_items = self.criterion(preds, batch)

        # Add AR mapping loss if descriptive embeddings are loaded
        mapping_loss = self.get_ar_mapping_loss()
        if mapping_loss.item() > 0:
            weight = getattr(self, "mapping_loss_weight", 0.1)
            weighted_mapping_loss = mapping_loss * weight
            loss = loss + weighted_mapping_loss

        return loss, loss_items
    

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes using CLIPTextEncoder for offline inference.

        Encodes class name text descriptions into CLIP embeddings and stores them
        for inference without needing the CLIP model at runtime.
        Also pushes class embeddings to all AR modules.

        Args:
            text (list[str]): List of class name strings.
            batch (int): Batch size for text encoding. Default: 80.
            cache_clip_model (bool): Whether to keep CLIP model loaded. Default: True.
        """
        print("Set classes using CLIPTextEncoder for offline inference")
        device = next(self.parameters()).device
        self.clip_text_encoder.load_model(device)
        txt_feats = self.clip_text_encoder.encode_batch(text, device=device, batch_split=batch)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)
        # Push class embeddings to AR modules directly
        self._sync_class_to_ar()
        if not cache_clip_model:
            self.clip_text_encoder.clip_model = None
            self.clip_text_encoder.tokenizer = None


class RGBTARDetModel(DetectionModel):
    """RGBT model with AR modules in neck and standard Detect head.

    Uses CLIP text embeddings to guide AR (AlignmentRegion) modules, while the final
    classification is still done by a standard Detect head.

    Current version is adapted for the simplified AR:
        - shared relation embeddings for all classes
        - shared RGB descriptive embeddings for all classes
        - shared IR descriptive embeddings for all classes
        - rel_class_map is optional compatibility info only
        - AR performs text-space enhancement and explicitly projects back to vision space

    Text feature flow:
        CLIPTextEncoder -> class_embeds (stored once, optional fallback)
        CLIPTextEncoder -> shared rel_embeds (optional)
        CLIPTextEncoder -> shared desc_rgb / desc_ir (optional)
            ↓
        AR(visual_feats, class_embeds, desc_rgb, desc_ir)
            ↓
        Detect(visual_feats) — standard fixed-channel classification
    """

    def __init__(self, cfg="yolov8-RGBT-earlyfusion-ar.yaml", ch=4, nc=None, verbose=True):
        """Initialize RGBTARDetModel with 4-channel input.

        Args:
            cfg (str | dict): Model config file path or dict.
            ch (int): Input channels (4 for RGBT). Default: 4.
            nc (int | None): Number of classes. Default: None.
            verbose (bool): Print model info. Default: True.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # placeholder for AR modules / class embeddings
        self.rel_feats = None
        self.desc_rgb_feats = None
        self.desc_ir_feats = None
        self.desc_rgb_class_map = None  # (N_desc_rgb,) long tensor mapping each desc to class idx
        self.desc_ir_class_map = None   # (N_desc_ir,) long tensor mapping each desc to class idx
        self.mapping_loss_weight = 0.1  # keep configurable
        self.align_loss_weight = 0.1    # Scheme A: L_align auxiliary loss weight
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def load(self, weights, verbose=True):
        """Load single-stream pretrained weights (e.g. ``yolov8s.pt``) into **both**
        the RGB and IR branches of this dual-stream RGBT model.

        Why the base ``BaseModel.load`` is insufficient
        -----------------------------------------------
        The stock ``BaseModel.load`` only does::

            csd = intersect_dicts(csd, self.state_dict())  # exact-key match
            self.load_state_dict(csd, strict=False)

        Single-stream YOLO checkpoints use keys like ``model.0.conv.weight`` starting
        at layer 0. This dual-stream RGBT YAML, however, starts with a ``Silence``
        (layer 0) + ``SilenceChannel`` (layer 1) preamble, so the real RGB backbone
        begins at layer **2**; the IR backbone begins at layer **12** (after a
        second ``SilenceChannel``). Neither branch's layer indices line up with the
        pretrained checkpoint, so ``intersect_dicts`` returns ~0 backbone matches
        and **both branches are effectively randomly initialized**.

        Observable symptom
        ------------------
        Because both branches are random, the AR module sees two random projections
        feeding ``proj_rgb`` and ``proj_ir``. Depending on the RNG state, one side
        can happen to produce a "collapsed" projection (all positions pointing in
        one direction, ``raw_class_mean`` ≈ 0.4) while the other produces a
        "dispersed" projection (near-orthogonal to class emb, ``raw_class_mean``
        ≈ 0.08). This shows up in training debug as::

            rgb_raw_class_mean: 0.08  →  rgb_confident_ratio: 0.02
            ir_raw_class_mean:  0.44  →  ir_confident_ratio:  0.40

        Fix
        ---
        We explicitly remap each pretrained ``model.N.*`` key to **two** RGBT
        layer indices — the RGB twin at ``N + rgb_offset`` and the IR twin at
        ``N + ir_offset``. Offsets are inferred from the YAML by locating the
        two ``SilenceChannel`` modules (one per branch).

        Special case: the very first conv of the IR branch has ``in_channels=1``
        (grayscale) whereas the pretrained first conv has ``in_channels=3`` (RGB).
        We collapse the three pretrained input channels to one by averaging,
        which preserves the learned spatial filter pattern.

        Fallback
        --------
        If the YAML layout does not contain two ``SilenceChannel`` modules (e.g.
        single-stream AR model or an exotic fusion topology), we gracefully fall
        back to the stock ``intersect_dicts`` behaviour.
        """
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        own_sd = self.state_dict()

        # ------------------------------------------------------------------
        # Step 0: detect branch structure by scanning the assembled model
        # for SilenceChannel sentinels. Two sentinels => dual-stream RGBT;
        # otherwise fall back to the stock loader.
        # ------------------------------------------------------------------
        silence_channel_idxs = []
        try:
            for i, layer in enumerate(self.model):
                if type(layer).__name__ == "SilenceChannel":
                    silence_channel_idxs.append(i)
        except Exception:
            silence_channel_idxs = []

        if len(silence_channel_idxs) < 2:
            # Not a dual-stream layout: stock behaviour.
            csd_std = intersect_dicts(csd, own_sd)
            self.load_state_dict(csd_std, strict=False)
            if verbose:
                LOGGER.info(
                    f"RGBTARDetModel: dual-stream layout not detected; "
                    f"transferred {len(csd_std)}/{len(own_sd)} items from pretrained "
                    f"(stock intersect_dicts behaviour)"
                )
            return

        # First real RGB conv index (right after the first SilenceChannel) and
        # first real IR conv index (right after the second SilenceChannel).
        # The pretrained checkpoint's layer 0 corresponds to the **first real
        # conv** of the single-stream reference backbone — i.e. our RGB layer
        # ``silence_channel_idxs[0] + 1``.
        rgb_first_layer = silence_channel_idxs[0] + 1
        ir_first_layer = silence_channel_idxs[1] + 1

        # We only remap backbone-range keys. The pretrained backbone typically
        # spans layers 0..9 in a single-stream yolov8s checkpoint; this RGBT
        # YAML has exactly (ir_first_layer - rgb_first_layer - 1) backbone
        # layers per branch (the -1 accounts for the IR SilenceChannel that
        # sits between the two backbones). Any pretrained key outside that
        # range (head, SPPF, detect, etc.) is handed to the stock loader with
        # its own indices and only matches RGBT layers that happen to share
        # the same absolute index (neck / fusion / SPPF / head keys after the
        # two backbones, which often do line up for mid-fusion YAMLs).
        backbone_len = ir_first_layer - rgb_first_layer - 1  # e.g. 9 for yolov8s
        backbone_pretrained_range = range(0, backbone_len)

        remapped_rgb = {}
        remapped_ir = {}
        first_conv_rgb_loaded = 0
        first_conv_ir_adapted = 0

        def _try_assign(target_dict, ir_key_flag, k_remapped, src_tensor):
            """Assign src_tensor to target_dict[k_remapped] if shape matches
            own_sd; handle the IR first-conv 3→1 channel adaptation."""
            nonlocal first_conv_ir_adapted, first_conv_rgb_loaded
            if k_remapped not in own_sd:
                return False
            dst_shape = own_sd[k_remapped].shape
            if src_tensor.shape == dst_shape:
                target_dict[k_remapped] = src_tensor
                if (not ir_key_flag) and k_remapped.endswith(".conv.weight") \
                        and src_tensor.dim() == 4 and src_tensor.shape[1] == 3:
                    first_conv_rgb_loaded += 1
                return True
            # IR first conv adaptation: 3-channel pretrained → 1-channel IR
            if (
                ir_key_flag
                and src_tensor.dim() == 4
                and dst_shape[1] == 1
                and src_tensor.shape[1] == 3
                and src_tensor.shape[0] == dst_shape[0]
                and src_tensor.shape[2:] == dst_shape[2:]
            ):
                target_dict[k_remapped] = src_tensor.mean(dim=1, keepdim=True)
                first_conv_ir_adapted += 1
                return True
            return False

        # ------------------------------------------------------------------
        # Step 1: remap backbone-range pretrained keys to both branches.
        # ------------------------------------------------------------------
        for k, v in csd.items():
            if not k.startswith("model."):
                continue
            parts = k.split(".")
            if len(parts) < 3:
                continue
            try:
                pre_idx = int(parts[1])
            except ValueError:
                continue
            if pre_idx not in backbone_pretrained_range:
                continue

            # RGB twin
            rgb_parts = parts.copy()
            rgb_parts[1] = str(pre_idx + rgb_first_layer)
            _try_assign(remapped_rgb, False, ".".join(rgb_parts), v)

            # IR twin
            ir_parts = parts.copy()
            ir_parts[1] = str(pre_idx + ir_first_layer)
            _try_assign(remapped_ir, True, ".".join(ir_parts), v)

        # ------------------------------------------------------------------
        # Step 2: for non-backbone keys (head / neck / SPPF / Detect), fall
        # back to exact-key intersection against own_sd. This recovers any
        # post-backbone layers whose absolute indices happen to line up.
        # ------------------------------------------------------------------
        remainder = {
            k: v for k, v in csd.items()
            if not (k.startswith("model.") and len(k.split(".")) >= 3
                    and k.split(".")[1].isdigit()
                    and int(k.split(".")[1]) in backbone_pretrained_range)
        }
        remainder_intersected = intersect_dicts(remainder, own_sd)

        # ------------------------------------------------------------------
        # Step 3: combined load. The three dicts have disjoint keys by
        # construction (RGB-branch, IR-branch, and non-backbone post-fusion
        # layers).
        # ------------------------------------------------------------------
        full_load = {**remapped_rgb, **remapped_ir, **remainder_intersected}
        self.load_state_dict(full_load, strict=False)

        # ------------------------------------------------------------------
        # Step 4: reset IR branch BatchNorm running statistics.
        #
        # Why: the IR branch conv weights are useful initialisations (they are
        # RGB pretrained filters, or 3→1 averaged for the first conv), but the
        # RGB pretrained BN running_mean / running_var encode RGB-image pixel
        # statistics. Feeding 1-channel thermal / grayscale IR data through
        # these RGB-tuned BNs systematically shifts features in the "wrong"
        # direction in CLIP space, producing the ``ir_raw_class_mean ≈ -0.3``
        # (anti-aligned with every class embedding) behaviour observed
        # empirically.
        #
        # Resetting to the PyTorch default (mean=0, var=1, tracked=0) lets
        # each IR BN freely adapt to the actual IR activation distribution
        # during training, avoiding the systematic sign inversion. The affine
        # parameters (``weight``, ``bias``) are kept as-loaded so the learned
        # scale/shift is preserved.
        # ------------------------------------------------------------------
        import torch.nn as _nn
        ir_layer_range = range(ir_first_layer, ir_first_layer + backbone_len)
        bn_reset_count = 0
        try:
            for ir_idx in ir_layer_range:
                if ir_idx >= len(self.model):
                    break
                ir_layer = self.model[ir_idx]
                for sub in ir_layer.modules():
                    if isinstance(sub, (_nn.BatchNorm2d, _nn.BatchNorm1d,
                                        _nn.BatchNorm3d, _nn.SyncBatchNorm)):
                        if sub.running_mean is not None:
                            sub.running_mean.zero_()
                        if sub.running_var is not None:
                            sub.running_var.fill_(1.0)
                        if sub.num_batches_tracked is not None:
                            sub.num_batches_tracked.zero_()
                        bn_reset_count += 1
        except Exception as _e:
            LOGGER.warning(
                f"RGBTARDetModel.load: failed to reset IR BN running stats: {_e!r}"
            )

        if verbose:
            LOGGER.info(
                f"RGBTARDetModel.load: dual-stream remap: "
                f"RGB branch loaded {len(remapped_rgb)} keys "
                f"(first-conv RGB: {first_conv_rgb_loaded}), "
                f"IR branch loaded {len(remapped_ir)} keys "
                f"(first-conv 3→1 averaged: {first_conv_ir_adapted}), "
                f"post-fusion intersect: {len(remainder_intersected)} keys, "
                f"IR BN running-stats reset: {bn_reset_count}. "
                f"rgb_first_layer={rgb_first_layer}, ir_first_layer={ir_first_layer}."
            )

    def set_classes(self, text, batch=80):
        """Encode class names with CLIP and store embeddings in all AR modules.

        Called once at training start (on_pretrain_routine_end). AR modules can use
        stored class embeddings as fallback / class-only mode.

        Args:
            text (list[str]): List of class name strings.
            batch (int): Batch size for CLIP encoding. Default: 80.
        """
        from ultralytics.nn.modules.clip_text_encoder import CLIPTextEncoder

        device = next(self.parameters()).device
        encoder = CLIPTextEncoder(model_name="ViT-B/32", embed_dim=512)
        encoder.load_model(device)
        txt_feats = encoder.encode_batch(text, device=device, batch_split=batch)

        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        self.model[-1].nc = len(text)

        # Push class embeddings into all AR modules
        count = self._sync_class_to_ar()

        LOGGER.info(
            f"RGBTARDetModel: set {len(text)} classes, stored in {count} AR module(s)"
        )

    def _sync_class_to_ar(self):
        """Push the currently-stored class embeddings (``self.txt_feats``)
        into every AR module in the model.

        Returns:
            int: number of AR modules that received the embeddings.
        """
        count = 0
        for m in self.model.modules():
            if isinstance(m, AR):
                m.set_class_embeddings(self.txt_feats)
                count += 1
        return count

    def load_class_embeddings(self, path):
        """Load pre-computed class embeddings from a ``.pt`` file and push
        them into every AR module.

        This is the offline counterpart of :meth:`set_classes`. It bypasses
        the CLIP text encoder entirely so the embeddings can come from the
        offline dataset-adaptation pipeline (see
        ``tools/offline_embedding_adapter.py``).

        The file may contain either:

        * a plain ``Tensor`` of shape ``(nc, C)`` or ``(1, nc, C)``, or
        * a ``dict`` with key ``embeddings`` holding such a tensor.

        The ``nc`` dimension must match the current head's class count.
        """
        data = torch.load(str(path), map_location="cpu")
        if isinstance(data, dict):
            feats = data.get("embeddings", None)
            if feats is None:
                raise ValueError(
                    f"[AR] class embedding file {path} is a dict but has no 'embeddings' key"
                )
        else:
            feats = data
        if not torch.is_tensor(feats):
            raise TypeError(f"[AR] unsupported class embedding payload in {path}: {type(feats)}")
        feats = feats.float()
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # (1, nc, C)
        if feats.dim() != 3:
            raise ValueError(
                f"[AR] class embedding tensor must be 2D or 3D, got shape {tuple(feats.shape)}"
            )

        expected_nc = getattr(self.model[-1], "nc", feats.shape[1])
        if feats.shape[1] != expected_nc:
            raise ValueError(
                f"[AR] class embedding count mismatch: file has {feats.shape[1]} classes but "
                f"head expects {expected_nc}"
            )

        self.txt_feats = feats
        self.model[-1].nc = feats.shape[1]
        count = self._sync_class_to_ar()
        LOGGER.info(
            f"[AR] Loaded adapted class embeddings from {path}: "
            f"shape={tuple(feats.shape)}, synced to {count} AR module(s)"
        )

    def _sync_rel_to_ar(self):
        """Push shared relation embeddings into all AR modules."""
        if self.rel_feats is None:
            return
        count = 0
        for m in self.model.modules():
            if isinstance(m, AR):
                try:
                    m.set_relational_embeddings(self.rel_feats)
                except TypeError:
                    # backward compatibility
                    m.set_relational_embeddings(self.rel_feats, None)
                count += 1
        LOGGER.info(
            f"Synced shared relational embeddings to {count} AR module(s): "
            f"{tuple(self.rel_feats.shape)}"
        )

    def _sync_desc_to_ar(self):
        """Push shared descriptive embeddings into all AR modules."""
        if self.desc_rgb_feats is None or self.desc_ir_feats is None:
            return
        count = 0
        for m in self.model.modules():
            if isinstance(m, AR):
                m.set_desc_embeddings(self.desc_rgb_feats, self.desc_ir_feats)
                count += 1
        LOGGER.info(
            f"Synced shared descriptive embeddings to {count} AR module(s): "
            f"RGB={tuple(self.desc_rgb_feats.shape)}, IR={tuple(self.desc_ir_feats.shape)}"
        )
        # also sync class maps if available
        self._sync_desc_class_map_to_ar()

    def _sync_desc_class_map_to_ar(self):
        """Push desc->class maps into all AR modules."""
        if self.desc_rgb_class_map is None or self.desc_ir_class_map is None:
            return
        count = 0
        for m in self.model.modules():
            if isinstance(m, AR):
                if hasattr(m, 'set_desc_class_map'):
                    m.set_desc_class_map(self.desc_rgb_class_map, self.desc_ir_class_map)
                    count += 1
        LOGGER.info(
            f"Synced desc class maps to {count} AR module(s): "
            f"RGB={tuple(self.desc_rgb_class_map.shape)}, IR={tuple(self.desc_ir_class_map.shape)}"
        )

    def set_shared_embeddings(self, rel_embeds=None, desc_rgb=None, desc_ir=None,
                               desc_rgb_class_map=None, desc_ir_class_map=None):
        """Directly set shared embeddings and sync them into all AR modules.

        This is convenient for trainer callback usage.
        """
        if rel_embeds is not None:
            if rel_embeds.dim() == 2:
                rel_embeds = rel_embeds.unsqueeze(0)
            self.rel_feats = rel_embeds
            self._sync_rel_to_ar()

        if desc_rgb is not None:
            if desc_rgb.dim() == 2:
                desc_rgb = desc_rgb.unsqueeze(0)
            self.desc_rgb_feats = desc_rgb

        if desc_ir is not None:
            if desc_ir.dim() == 2:
                desc_ir = desc_ir.unsqueeze(0)
            self.desc_ir_feats = desc_ir

        if desc_rgb_class_map is not None:
            self.desc_rgb_class_map = desc_rgb_class_map
        if desc_ir_class_map is not None:
            self.desc_ir_class_map = desc_ir_class_map

        if self.desc_rgb_feats is not None and self.desc_ir_feats is not None:
            self._sync_desc_to_ar()

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Forward pass with text features routed to AR modules."""
        # 获取实际模型（处理 DDP 包装）
        real_model = self.module if hasattr(self, 'module') else self

        # 获取类嵌入（txt_feats）
        txt_feats = real_model.txt_feats
        if txt_feats is None:
            # 如果未设置，使用占位符（仅用于避免错误）
            nc = getattr(real_model.model[-1], 'nc', 80)
            txt_feats = torch.randn(1, nc, 512, device=x.device, dtype=x.dtype)
        txt_feats = txt_feats.to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)

        # 获取描述性嵌入
        desc_rgb = real_model.desc_rgb_feats
        desc_ir = real_model.desc_ir_feats
        if desc_rgb is not None:
            desc_rgb = desc_rgb.to(device=x.device, dtype=x.dtype)
            if len(desc_rgb) != len(x):
                desc_rgb = desc_rgb.repeat(len(x), 1, 1)
        if desc_ir is not None:
            desc_ir = desc_ir.to(device=x.device, dtype=x.dtype)
            if len(desc_ir) != len(x):
                desc_ir = desc_ir.repeat(len(x), 1, 1)

        y, dt, embeddings = [], [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)

            if isinstance(m, AR):
                x = m(x, txt_feats, desc_rgb, desc_ir)
            elif isinstance(m, WorldDetect):
                # Route class embeddings into WorldDetect's contrastive classifier.
                # Works with both standard Detect YAMLs (no-op, this branch never triggers)
                # and World YAMLs where the last layer is WorldDetect.
                x = m(x, txt_feats)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                embeddings.append(pooled)
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        return x

    def loss(self, batch, preds=None):
        """Standard detection loss + AR mapping loss.

        No per-batch txt_feats in batch needed — AR uses stored embeddings.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])

        loss, loss_items = self.criterion(preds, batch)

        # Add AR mapping loss if descriptive embeddings are loaded
        mapping_loss = self._get_ar_mapping_loss()
        if mapping_loss is not None and torch.isfinite(mapping_loss) and mapping_loss.item() > 0:
            weight = getattr(self, "mapping_loss_weight", 0.1)
            loss = loss + mapping_loss * weight

        # Scheme A: auxiliary alignment loss. Pulls proj_rgb / proj_ir toward
        # the nearest class embedding so the projections can mature during the
        # AR warmup window (when the AR rewrite path is disabled).
        align_loss = self._get_ar_align_loss()
        if align_loss is not None and torch.isfinite(align_loss) and align_loss.item() > 0:
            weight = getattr(self, "align_loss_weight", 0.1)
            loss = loss + align_loss * weight

        # Guard: if loss is NaN/Inf, skip this step to avoid corrupting all parameters
        if not torch.isfinite(loss):
            LOGGER.warning("[AR] Loss is NaN/Inf, returning zero loss for this step")
            loss = torch.zeros_like(loss, requires_grad=True)

        return loss, loss_items

    def _get_ar_mapping_loss(self):
        """Collect cross-modal mapping losses from all AR modules."""
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0

        for m in self.model.modules():
            if isinstance(m, AR):
                if hasattr(m, "get_mapping_loss"):
                    loss_val = m.get_mapping_loss()
                else:
                    loss_val = getattr(m, "mapping_loss", None)

                if loss_val is not None:
                    total = total + loss_val
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total

    def _get_ar_align_loss(self):
        """Collect L_align auxiliary losses from all AR modules (Scheme A)."""
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0

        for m in self.model.modules():
            if isinstance(m, AR):
                loss_val = getattr(m, "align_loss", None)
                if loss_val is not None:
                    total = total + loss_val
                    count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total

    def _extract_desc_and_map(self, data):
        """Extract embeddings and optional class_map from loaded .pt data.

        Supports two formats:
            - plain tensor: (N, C) or (1, N, C)
            - dict: {'embeddings': tensor, 'class_map': tensor}
        """
        if isinstance(data, dict):
            emb = data['embeddings']
            cmap = data.get('class_map', None)
        else:
            emb = data
            cmap = None
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        return emb, cmap

    def load_descriptive_embeddings(self, rgb_path, ir_path):
        """Load pre-computed SHARED descriptive embeddings for AR modules.

        Each .pt file can be either a plain tensor or a dict with keys
        'embeddings' and 'class_map'.

        Args:
            rgb_path (str): Path to shared RGB descriptive embeddings .pt file.
            ir_path (str): Path to shared IR descriptive embeddings .pt file.
        """
        raw_rgb = torch.load(rgb_path, map_location="cpu")
        raw_ir = torch.load(ir_path, map_location="cpu")

        desc_rgb, rgb_cmap = self._extract_desc_and_map(raw_rgb)
        desc_ir, ir_cmap = self._extract_desc_and_map(raw_ir)

        # 先删除旧的普通属性，再注册为 buffer
        if hasattr(self, 'desc_rgb_feats'):
            delattr(self, 'desc_rgb_feats')
        if hasattr(self, 'desc_ir_feats'):
            delattr(self, 'desc_ir_feats')

        self.register_buffer('desc_rgb_feats', desc_rgb)
        self.register_buffer('desc_ir_feats', desc_ir)

        if rgb_cmap is not None:
            self.desc_rgb_class_map = rgb_cmap
        if ir_cmap is not None:
            self.desc_ir_class_map = ir_cmap

        self._sync_desc_to_ar()
        LOGGER.info(f"Loaded descriptive embeddings: RGB={desc_rgb.shape}, IR={desc_ir.shape}")
        if rgb_cmap is not None:
            LOGGER.info(f"  with class maps: RGB={rgb_cmap.shape}, IR={ir_cmap.shape if ir_cmap is not None else None}")

    def load_relational_embeddings(self, rel_path, rel_class_map_path=None):
        """Load pre-computed SHARED relational embeddings for AR modules."""
        rel_embeds = torch.load(rel_path, map_location="cpu")
        if rel_embeds.dim() == 2:
            rel_embeds = rel_embeds.unsqueeze(0)

        self.rel_feats = rel_embeds
        self._sync_rel_to_ar()

        LOGGER.info(f"Loaded relational embeddings: {rel_embeds.shape} from {rel_path}")

    def load_desc_class_maps(self, rgb_map_path, ir_map_path):
        """Load desc->class maps from separate .pt files.

        Args:
            rgb_map_path (str): Path to RGB desc class map .pt file (long tensor).
            ir_map_path (str): Path to IR desc class map .pt file (long tensor).
        """
        self.desc_rgb_class_map = torch.load(rgb_map_path, map_location="cpu").long()
        self.desc_ir_class_map = torch.load(ir_map_path, map_location="cpu").long()
        self._sync_desc_class_map_to_ar()
        LOGGER.info(
            f"Loaded desc class maps: RGB={self.desc_rgb_class_map.shape}, "
            f"IR={self.desc_ir_class_map.shape}"
        )

class Ensemble(torch.nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Example:
    ```python
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    ```

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))

    threshold = None
    scale = d.get("scale", None)

    if scales:
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        if len(scales[scale]) == 3:
            depth, width, max_channels = scales[scale]
        elif len(scales[scale]) == 4:
            depth, width, max_channels, threshold = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            YOLOv4_BottleneckCSP,
            YOLOv4_Bottleneck,
            C1,
            C2,
            C2f,
            C3k2,
            C3k2_DeepDBB,
            C3k2_DBB,
            C3k2_WDBB,
            C2f_DeepDBB,
            C2f_WDBB,
            C2f_DBB,
            C3k_RDBB,
            C2f_RDBB,
            C3k2_RDBB,
            A2C2f,
            DSC3k2,
            ConvNormLayer,
            BasicBlock,
            BottleNeck,
            MANet,
            MANet_FasterBlock,
            MANet_FasterCGLU,
            MANet_Star,
            RepNCSPELAN4,
            ELAN1,
            ELAN,
            ELAN_H,
            ELAN_t,
            SPPCSPCSIM,
            SPPCSPC,
            MP_1,
            MP_2,
            RepConv,
            DSConv,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            AR,
            AlignmentRegion,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
        }
    )

    repeat_modules = frozenset(
        {
            BottleneckCSP,
            YOLOv4_BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C3k2_DeepDBB,
            C3k2_DBB,
            C3k2_WDBB,
            C2f_DeepDBB,
            C2f_WDBB,
            C2f_DBB,
            C3k_RDBB,
            C2f_RDBB,
            C3k2_RDBB,
            A2C2f,
            DSC3k2,
            MANet,
            MANet_FasterBlock,
            MANet_FasterCGLU,
            MANet_Star,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
        }
    )

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        if m in (AR, AlignmentRegion):
            if not isinstance(f, list) or len(f) != 2:
                raise ValueError(f"AR expects two input indices before fusion, but got f={f}")

            c1_rgb, c1_ir = ch[f[0]], ch[f[1]]
            if c1_rgb != c1_ir:
                raise ValueError(
                    f"AR requires equal channel dims before fusion, got rgb={c1_rgb}, ir={c1_ir}"
                )

            # yaml args example:
            # [c2, n, ec, nh, gc, shortcut, g, e, threshold, ...]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1_rgb, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1

        elif m in base_modules:
            c1, c2 = ch[f], args[0]

            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            # 修复这里：原来写成 if m in (C2fAttn) 会报 type is not iterable
            if m in (C2fAttn,):
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                    if args[2] > 1 else args[2]
                )

            args = [c1, c2, *args[1:]]

            if m in repeat_modules:
                args.insert(2, n)
                n = 1

            if m in (C3k2,):
                legacy = False
                if scale in "mlx":
                    args[3] = True

            if m in (A2C2f,) and scale in "lx":
                print("!!!")
                args.extend((True, 1.2))

        elif m is AIFI:
            args = [ch[f], *args]

        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1

        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4

        elif m is Blocks:
            block_type = globals()[args[1]]
            c1, c2 = ch[f], args[0] * block_type.expansion
            args = [c1, args[0], block_type, *args[2:]]

        elif m in [CSPResNet_CBS, CSPResNet, ConvBNLayer, ResSPP, CoordConv]:
            c2 = args[1]

        elif m in [ResNet50vd, ResNet50vd_dcn, ResNet101vd, PPConvBlock, Res2net50]:
            c2 = args[0]

        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m is ADD:
            c2 = max(ch[x] for x in f)

        elif m is CrossAttentionShared:
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c2]

        elif m is GPT:
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c2]

        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]

        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]

        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]

        elif m in frozenset({CrossC2f, CrossC3k2}):
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            c1 = ch[f[0]]
            args = [c1, c2, n, *args[1:]]
            if scale in "mlx":
                args[3] = True
            n = 1

        elif m in [CBH, ES_Bottleneck, DWConvblock]:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]

        elif m in frozenset({ZeroConv2d, ZeroConv1d}):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                if c2 >= 8:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]

        elif m in frozenset({CrossMLCA, CrossMLCAv2}):
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c2]

        elif m in {
            EMA,
            BiLevelRoutingAttention,
            BiLevelRoutingAttention_nchw,
            TripletAttention,
            CoordAtt,
            CBAM,
            BAMBlock,
            LSKBlock,
            SEAttention,
            CPCA,
            FocalModulation,
            EfficientAttention,
            MPCA,
            deformable_LKA,
            EffectiveSEModule,
            LSKA,
            SegNext_Attention,
            DAttention,
            MLCA,
            FocusedLinearAttention,
            LocalWindowAttention,
            CAA,
            ELA,
            AFGCAttention,
        }:
            c2 = ch[f]
            args = [c2, *args]

        elif m in {SimAM}:
            c2 = ch[f]

        elif m is HyperComputeModule:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, threshold]

        elif m is TensorSelector:
            c2 = args[1]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [args[0]]
            n = 1

        elif m is CrossTransformerFusion:
            c1 = ch[f[0]]
            c2 = sum(ch[x] for x in f)
            args = [c1, *args]

        elif m is HyperACE:
            c1 = ch[f[1]]
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            he = args[1]
            if scale in "n":
                he = int(args[1] * 0.5)
            elif scale in "x":
                he = int(args[1] * 1.5)
            args = [c1, c2, n, he, *args[2:]]
            n = 1
            if scale in "lx":
                args.append(False)

        elif m is DownsampleConv:
            c1 = ch[f]
            c2 = c1 * 2
            args = [c1]
            if scale in "lx":
                args.append(False)
                c2 = c1

        elif m is FullPAD_Tunnel:
            c2 = ch[f[0]]

        elif m in ((WorldDetect, ImagePoolingAttn) + DETECT_CLASS + V10_DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS):
            args.append([ch[x] for x in f])

            if m in SEGMENT_CLASS:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                if m in (Segment_LSCD,):
                    args[3] = make_divisible(min(args[3], max_channels) * width, 8)

            if m in (Detect_LSCD,):
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)

            if m in (Pose_LSCD, OBB_LSCD):
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)

            if m in DETECT_CLASS + V10_DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS:
                m.legacy = legacy

        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])

        elif m is CBLinear:
            c2 = make_divisible(min(args[0][-1], max_channels) * width, 8)
            c1 = ch[f]
            args = [c1, [make_divisible(min(c2_, max_channels) * width, 8) for c2_ in args[0]], *args[1:]]

        elif m is CBFuse:
            c2 = ch[f[-1]]

        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]

        elif m is SilenceChannel:
            c2 = args[1] - args[0]

        elif m is NumberToChannel:
            c2 = ch[f] * 2

        elif m is ChannelToNumber:
            c2 = 3

        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t

        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)

def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa, returns n, s, m, l, or x
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if m == "segment":
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect,DetectDeepDBB,DetectV8,DetectAux,DetectWDBB, WorldDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect