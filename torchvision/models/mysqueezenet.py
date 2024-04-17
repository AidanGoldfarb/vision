from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init

import time

from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = ["mySqueezeNet", "mySqueezeNet1_0_Weights", "mySqueezeNet1_1_Weights", "mysqueezenet1_0", "mysqueezenet1_1"]


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class mySqueezeNet(nn.Module):
    def __init__(self, timed,sync,version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
            
        else:
            # FIXME: Is this needed? mySqueezeNet should only be called from the
            # FIXME: mysqueezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported mySqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if sync:
            torch.cuda.synchronize()
        # x = self.features(x)
        # x = self.classifier(x)
        # return torch.flatten(x, 1)
        times = []

        # st = time.perf_counter_ns()
        # #x = self.features(x)
        # x = self.featurescomp(x)
        # et = time.perf_counter_ns()
        # times.append(et-st)

        for module in self.features:
            if timed:
                st = time.perf_counter_ns()
            x = module(x)
            torch.cuda.synchronize()
            if timed:
                et = time.perf_counter_ns()
                times.append(et-st)

        if timed:
            st = time.perf_counter_ns()
        x = self.classifier(x)
        if sync:
            torch.cuda.synchronize()
        if timed:
            et = time.perf_counter_ns()
            times.append(et-st)

        if timed:
            st = time.perf_counter_ns()
        x = torch.flatten(x,1)
        if sync:
            torch.cuda.synchronize()
        if timed:
            et = time.perf_counter_ns()
            times.append(et-st)

        return x,times


def _mysqueezenet(
    version: str,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> mySqueezeNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = mySqueezeNet(version, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/49#issuecomment-277560717",
    "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
}


class mySqueezeNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (21, 21),
            "num_params": 1248424,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 58.092,
                    "acc@5": 80.420,
                }
            },
            "_ops": 0.819,
            "_file_size": 4.778,
        },
    )
    DEFAULT = IMAGENET1K_V1


class mySqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (17, 17),
            "num_params": 1235496,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 58.178,
                    "acc@5": 80.624,
                }
            },
            "_ops": 0.349,
            "_file_size": 4.729,
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_model()
@handle_legacy_interface(weights=("pretrained", mySqueezeNet1_0_Weights.IMAGENET1K_V1))
def mysqueezenet1_0(
    *, weights: Optional[mySqueezeNet1_0_Weights] = None, progress: bool = True, **kwargs: Any
) -> mySqueezeNet:
    """mySqueezeNet model architecture from the `mySqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        weights (:class:`~torchvision.models.mySqueezeNet1_0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.mySqueezeNet1_0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mysqueezenet.mySqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.mySqueezeNet1_0_Weights
        :members:
    """
    weights = mySqueezeNet1_0_Weights.verify(weights)
    return _mysqueezenet("1_0", weights, progress, **kwargs)


@register_model()
@handle_legacy_interface(weights=("pretrained", mySqueezeNet1_1_Weights.IMAGENET1K_V1))
def mysqueezenet1_1(
    *, weights: Optional[mySqueezeNet1_1_Weights] = None, progress: bool = True, **kwargs: Any
) -> mySqueezeNet:
    """mySqueezeNet 1.1 model from the `official mySqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.

    mySqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than mySqueezeNet 1.0, without sacrificing accuracy.

    Args:
        weights (:class:`~torchvision.models.mySqueezeNet1_1_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.mySqueezeNet1_1_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mysqueezenet.mySqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.mySqueezeNet1_1_Weights
        :members:
    """
    weights = mySqueezeNet1_1_Weights.verify(weights)
    return _mysqueezenet("1_1", weights, progress, **kwargs)
