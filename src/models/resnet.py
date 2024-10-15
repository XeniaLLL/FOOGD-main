# import torch
# import torchvision
#
#
# class ResNet18(torch.nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.model = torchvision.models.resnet18(num_classes=num_classes)
#         self.fc = self.model.fc
#
#     def forward(self, x):
#         return self.model.forward(x)
#
#     def intermediate_forward(self, x):
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
#
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#
#         return x

from typing import Any, Optional, Callable, List

import torch
from torch import nn, Tensor

# from src.model.grad_batchnorm import GradBatchNorm2d


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    """add has_bn"""
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_bn:
            self.bn1 = norm_layer(planes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """add has_bn and bn_block_num"""

    def __init__(
            self,
            block: BasicBlock,
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn: bool = True,
            bn_block_num=4,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], has_bn=has_bn and (0 < bn_block_num))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       has_bn=has_bn and (1 < bn_block_num))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       has_bn=has_bn and (2 < bn_block_num))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       has_bn=has_bn and (3 < bn_block_num))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block: BasicBlock,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            has_bn: bool = True,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                has_bn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    has_bn=has_bn,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def intermediate_forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    # def change_bn(self):
    #     self.bn1 = GradBatchNorm2d(self.bn1)
    #
    #     self.layer1[0].bn1 = GradBatchNorm2d(self.layer1[0].bn1)
    #     self.layer1[0].bn2 = GradBatchNorm2d(self.layer1[0].bn2)
    #     self.layer1[1].bn1 = GradBatchNorm2d(self.layer1[1].bn1)
    #     self.layer1[1].bn2 = GradBatchNorm2d(self.layer1[1].bn2)
    #
    #     self.layer2[0].bn1 = GradBatchNorm2d(self.layer2[0].bn1)
    #     self.layer2[0].bn2 = GradBatchNorm2d(self.layer2[0].bn2)
    #     self.layer2[0].downsample[1] = GradBatchNorm2d(self.layer2[0].downsample[1])
    #     self.layer2[1].bn1 = GradBatchNorm2d(self.layer2[1].bn1)
    #     self.layer2[1].bn2 = GradBatchNorm2d(self.layer2[1].bn2)
    #
    #     self.layer3[0].bn1 = GradBatchNorm2d(self.layer3[0].bn1)
    #     self.layer3[0].bn2 = GradBatchNorm2d(self.layer3[0].bn2)
    #     self.layer3[0].downsample[1] = GradBatchNorm2d(self.layer3[0].downsample[1])
    #     self.layer3[1].bn1 = GradBatchNorm2d(self.layer3[1].bn1)
    #     self.layer3[1].bn2 = GradBatchNorm2d(self.layer3[1].bn2)
    #
    #     self.layer4[0].bn1 = GradBatchNorm2d(self.layer4[0].bn1)
    #     self.layer4[0].bn2 = GradBatchNorm2d(self.layer4[0].bn2)
    #     self.layer4[0].downsample[1] = GradBatchNorm2d(self.layer4[0].downsample[1])
    #     self.layer4[1].bn1 = GradBatchNorm2d(self.layer4[1].bn1)
    #     self.layer4[1].bn2 = GradBatchNorm2d(self.layer4[1].bn2)
    #
    # def trainable_parameters(self):
    #     return [p for p in self.parameters() if p.requires_grad]
    #
    # def set_running_stat_grads(self):
    #     for m in self.modules():
    #         if isinstance(m, GradBatchNorm2d):
    #             m.set_running_stat_grad()
    #
    # def clip_bn_running_vars(self):
    #     for m in self.modules():
    #         if isinstance(m, GradBatchNorm2d):
    #             m.clip_bn_running_var()


def ResNet18(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

