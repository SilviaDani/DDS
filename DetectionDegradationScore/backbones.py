from enum import Enum
from typing import List, NamedTuple
from typing import Union


class LayerConfig(NamedTuple):
    indices: List[Union[int, str]]
    channels: List[int]


class Backbone(Enum):
    YOLO_V11_M = "yolov11m"
    VGG_16 = "vgg16"
    MOBILENET_V3_L = "mobilenetv3-large"
    EFFICIENTNET_V2_M = "efficientnetv2-m"
    FASTERRCNN_MOBILENET_V3_LARGE_FPN = "fasterrcnn_mobilenet_v3_large_fpn"

    @property
    def config(self) -> LayerConfig:
        """Get the layer configuration for the backbone."""
        configs = {
            Backbone.YOLO_V11_M: LayerConfig(
                indices=[9],
                channels=[512],
            ),
            Backbone.VGG_16: LayerConfig(
                indices=[23, 30],
                channels=[512, 512],
            ),
            Backbone.MOBILENET_V3_L: LayerConfig(
                indices=[5, 12],
                channels=[40, 112],
            ),
            Backbone.EFFICIENTNET_V2_M: LayerConfig(
                indices=[6, 7],
                channels=[304, 512],
            ),
            Backbone.FASTERRCNN_MOBILENET_V3_LARGE_FPN: LayerConfig(
                #indices=["0", "1", "pool"],  # Use string keys matching FPN outputs
                #channels=[256, 256, 256],
                indices=["0", "1"],
                channels=[256, 256],
            ),
        }
        return configs[self]
