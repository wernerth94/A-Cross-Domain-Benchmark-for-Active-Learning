'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from core.classifier import kaiming_uniform_seeded, SeededLinear

class SeededConv2d(nn.Conv2d):
    """
    Replaces the default kaiming initialization with a seeded version of itself
    """

    def __init__(self, model_rng, *args, **kwargs):
        self.model_rng = model_rng
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        kaiming_uniform_seeded(self.model_rng, self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                self.bias.uniform_(-bound, bound, generator=self.model_rng)
                # init.uniform_(self.bias, -bound, bound)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, model_rng, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SeededConv2d(model_rng, in_planes, planes,
                                  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SeededConv2d(model_rng, planes, planes, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeededConv2d(model_rng, in_planes, self.expansion*planes,
                             kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, model_rng, num_classes=10, in_channels=3, dropout=None, add_head=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = SeededConv2d(model_rng, in_channels, 64, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(model_rng, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(model_rng, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(model_rng, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(model_rng, block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if add_head:
            self.linear = SeededLinear(model_rng, 512*block.expansion, num_classes)

    def _make_layer(self, model_rng, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(model_rng, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _encode(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        out = self.layer2(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        out = self.layer3(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        out = self.layer4(out)
        out = self.avgpool(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self._encode(x)
        if hasattr(self, "linear"):
            out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
