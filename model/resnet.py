# -*- coding: utf-8 -*-
from model.layer import *

from .abc_model import RateModel
from experiment.cifar.config.config import args

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, expand=1, **kwargs_spikes):
        super(BasicBlock, self).__init__()
        self.expand = expand
        self.conv1 = nn.Conv2d(in_planes, planes * expand, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * expand)
        self.spike1 = LIFLayer(**kwargs_spikes)
        self.conv2 = nn.Conv2d(planes, planes * expand, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * expand)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion * expand, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes * expand)
            )
        self.spike2 = LIFLayer(**kwargs_spikes)

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.spike2(out)
        return out


def make_bn(module):
    module.ann_branch = True
    return module




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
class SequentialModule(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self._modules.values():
            input = module(input, **kwargs)
        return input

    def get_spike(self):
        spikes = []
        for module in self._modules.values():
            spikes_module = module.get_spike()
            spikes += spikes_module
        return spikes
class ResNet(RateModel):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, **kwargs_spikes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            LIFLayer(**kwargs_spikes)
        )
        self.layer1 = self._make_layer(block, 64, num_block_layers[0], stride=1, **kwargs_spikes)
        self.layer2 = self._make_layer(block, 128, num_block_layers[1], stride=2, **kwargs_spikes)
        self.layer3 = self._make_layer(block, 256, num_block_layers[2], stride=2, **kwargs_spikes)
        self.layer4 = self._make_layer(block, 512, num_block_layers[3], stride=2, **kwargs_spikes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
        )

        self.scala1 = nn.Sequential(
            DeepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            DeepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            DeepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs_spikes):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs_spikes))
            self.in_planes = planes * block.expansion
        return SequentialModule(*layers)

    def forward(self, x):
        if self.training and self.rate_prop:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[0]//args.T
        feature_list = [0.0, 0.0, 0.0, 0.0]

        out = self.conv0(x)
        out = self.layer1(out)
        if (self.rate_prop and self.training) :
            if not self.training:
                out1 = out.reshape(args.T, -1, *out.shape[1:])
                out1 = out1.mean(0)
            else:
                out1 = out
            feature_list[0] = out1
        out = self.layer2(out)
        if (self.rate_prop and self.training):
            if not self.training:
                out2 = out.reshape(args.T, -1, *out.shape[1:])
                out2 = out2.mean(0)
            else:
                out2 = out
            feature_list[1] = out2
        out = self.layer3(out)
        if (self.rate_prop and self.training):
            if not self.training:
                out3 = out.reshape(args.T, -1, *out.shape[1:])
                out3 = out3.mean(0)
            else:
                out3 = out
            feature_list[2] = out3
        out = self.layer4(out)
        out = self.avg_pool(out)
        out_feature = out.view(out.shape[0], -1)

        out = self.classifier(out_feature)

        if (self.rate_prop and self.training) :

            out1_feature = self.scala1(feature_list[0]).view(batch_size, -1)
            out2_feature = self.scala2(feature_list[1]).view(batch_size, -1)
            out3_feature = self.scala3(feature_list[2]).view(batch_size, -1)

            out1 = self.fc1(out1_feature)
            out2 = self.fc2(out2_feature)
            out3 = self.fc3(out3_feature)


            return [out1, out2, out3, out]
        else:
            return out







class ResNet19(RateModel):
    def __init__(self, block, num_block_layers, num_classes=10, in_channel=3, **kwargs_spikes):
        super(ResNet19, self).__init__()
        self.in_planes = 128
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel, self.in_planes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            LIFLayer(**kwargs_spikes)
        )
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], stride=1, **kwargs_spikes)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], stride=2, **kwargs_spikes)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], stride=2, **kwargs_spikes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * block.expansion, 256, bias=True),
            LIFLayer(**kwargs_spikes),
            nn.Linear(256, num_classes, bias=True),
        )

        self.scala1 = nn.Sequential(

            DeepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            #nn.AvgPool2d(4, 4)
            nn.AvgPool2d(6, 6)#dvs
        )

        self.scala2 = nn.Sequential(

            DeepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            #nn.AvgPool2d(4, 4)
            nn.AvgPool2d(6, 6)#dvs
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride, **kwargs_spikes):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs_spikes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training and self.rate_prop:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[0] // args.T
        feature_list = [0.0, 0.0]
        out = self.conv0(x)
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
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        out1, out2 = 0, 0
        if (self.rate_prop and self.training) or not self.training:

            out1_feature = self.scala1(feature_list[0]).view(batch_size, -1)
            out2_feature = self.scala2(feature_list[1]).view(batch_size, -1)
            out1 = self.fc1(out1_feature)
            out2 = self.fc2(out2_feature)
        return [out1, out2, out]


def resnet18(num_classes=10, in_channel=3, neuron_dropout=0.0, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channel=in_channel, **kwargs)


def resnet19(num_classes=10, in_channel=3, neuron_dropout=0.0, **kwargs):
    return ResNet19(BasicBlock, [3, 3, 2], num_classes, in_channel=in_channel, **kwargs)

