__all__ = [
    "resnet50_fpn_1x",
]

from icevision.imports import *
from icevision.models.mmdet.utils import *
from icevision.models.mmdet.models.double_heads.backbones.backbone_config import (
    MMDetDOUBLEHEADRCNNBackboneConfig,
)

base_config_path = mmdet_configs_path / "double_heads"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/double_heads"

resnet50_fpn_1x = MMDetDOUBLEHEADRCNNBackboneConfig(
    config_path=base_config_path / "dh_faster_rcnn_r50_fpn_1x_coco.py",
    weights_url=f"{base_weights_url}/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth",
)


