import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from spikingjelly.activation_based import neuron
from .abc_model import RateModel
from .layer import LIFLayer as MyLIFNode
from .layer import *
from experiment.imagenet.config.config import args

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)




class BasicBlock(nn.Module):
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
            neuron: callable = None,
            **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sn1 = neuron(**kwargs)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn2 = neuron(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x
        out = self.sn1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.sn2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


def make_bn(module):
    module.ann_branch = True
    return module


def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        make_bn(nn.BatchNorm2d(128)),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        make_bn(nn.BatchNorm2d(128)),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        make_bn(nn.BatchNorm2d(channel_out)),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class StdConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):

        super(StdConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = make_bn(nn.BatchNorm2d(channel_out, affine=affine))
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = make_bn(nn.BatchNorm2d(channel_out, affine=affine))

        if channel_in != channel_out:
            self.residual = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, bias=False)
            self.res_bn = make_bn(nn.BatchNorm2d(channel_out, affine=affine))
        else:
            self.residual = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual is not None:
            identity = self.residual(identity)
            identity = self.res_bn(identity)

        out += identity
        out = self.relu(out)
        return out

class DeepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(DeepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            make_bn(nn.BatchNorm2d(channel_in, affine=affine)),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            make_bn(nn.BatchNorm2d(channel_out, affine=affine)),
            nn.ReLU(inplace=False),
        )
        self.residual = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride, bias=False)
        self.res_bn = make_bn(nn.BatchNorm2d(channel_out, affine=affine))
        self.act = nn.ReLU(inplace=False)
    def forward(self, x):

        res_out = self.residual(x)
        res_out = self.res_bn(res_out)
        out = res_out + self.op(x)
        out = self.act(out)

        return out

class PreactSpikingResNet(RateModel):
    def __init__(
            self,
            block: Type[Union[BasicBlock, ]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            neuron: callable = None,
            **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # need_sync = dist.is_available() and dist.is_initialized()
            # if need_sync and dist.get_world_size(dist.group.WORLD) > 1:
            #     norm_layer = lambda *args, **kwargs: nn.BatchNorm2d(momentum = 0.1 / config.args.T, *args, **kwargs)
            # else:
            #     norm_layer = lambda *args, **kwargs: nn.BatchNorm2d(momentum = 0.1 / config.args.T, *args, **kwargs)
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       neuron=neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       neuron=neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       neuron=neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.scala1 = nn.Sequential(
            DeepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            StdConvBlock(
                channel_in=128* block.expansion,
                channel_out=128 * block.expansion
            ),
            DeepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),

            StdConvBlock(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            StdConvBlock(
                channel_in=512 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            DeepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            StdConvBlock(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            StdConvBlock(
                channel_in=512 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            StdConvBlock(
                channel_in=512 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )


        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ) and m.convNeuron3.gamma is not None:
                    nn.init.constant_(m.convNeuron3.gamma, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.convNeuron2.gamma is not None:
                    nn.init.constant_(m.convNeuron2.gamma, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock,]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            neuron: callable = None,
            **kwargs
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                neuron, **kwargs
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
                    neuron=neuron,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        if self.training and self.rate_prop:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[0]//args.T
        feature_list = [0.0, 0.0, 0.0, 0.0]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        if (self.rate_prop and self.training) or not self.training:
            if not self.training:
                out1 = out.reshape(args.T, -1, *out.shape[1:])
                out1 = out1.mean(0)
            else:
                out1 = out
            feature_list[0] = out1
        out = self.layer2(out)
        if (self.rate_prop and self.training) or not self.training:
            if not self.training:
                out2 = out.reshape(args.T, -1, *out.shape[1:])
                out2 = out2.mean(0)
            else:
                out2 = out
            feature_list[1] = out2

        out = self.layer3(out)
        if (self.rate_prop and self.training) or not self.training:
            if not self.training:
                out3 = out.reshape(args.T, -1, *out.shape[1:])
                out3 = out3.mean(0)
            else:
                out3 = out
            feature_list[2] = out3

        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if (self.rate_prop and self.training) or not self.training:

            out1_feature = self.scala1(feature_list[0]).view(batch_size, -1)
            out2_feature = self.scala2(feature_list[1]).view(batch_size, -1)
            out3_feature = self.scala3(feature_list[2]).view(batch_size, -1)

            out1 = self.fc1(out1_feature)
            out2 = self.fc2(out2_feature)
            out3 = self.fc3(out3_feature)
            return [out1, out2, out3, out]
        else:
            return out



    def get_spike(self):
        raise NotImplementedError('get_spike not implemented now!')


def _preact_spiking_resnet(arch, block, layers, pretrained, progress, single_step_neuron, **kwargs):
    model = PreactSpikingResNet(block, layers, neuron=single_step_neuron, **kwargs)

    return model

def preact_resnet18(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return _preact_spiking_resnet('resnet18', BasicBlock, [2,2,2,2], pretrained, progress,
                                  single_step_neuron=MyLIFNode, **kwargs)
def preact_resnet34(pretrained=False, progress=True, single_step_neuron: callable = None, **kwargs):
    return _preact_spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                                  single_step_neuron=MyLIFNode, **kwargs)

