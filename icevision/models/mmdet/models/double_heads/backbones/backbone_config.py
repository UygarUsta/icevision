from icevision.models.mmdet.utils import *


class MMDetDOUBLEHEADRCNNBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="double_heads", **kwargs)
