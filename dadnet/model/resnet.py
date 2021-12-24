"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dadnet.modules import Flatten
from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.fake_linear_layer import FakeLinear

FC1_OUT = 1024
FC2_OUT = 1024
FC3_OUT = 1024


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu3 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.relu4 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.relu4(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_shape, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_shape[1], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out


class ResNet18(ResNet):
    def __init__(self, in_shape, num_classes):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], in_shape, 512)


class ResNet34(ResNet):
    def __init__(self, in_shape, num_classes):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], in_shape, num_classes)


class ResNet50(ResNet):
    def __init__(self, in_shape, num_classes):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], in_shape, num_classes)


class ResNet101(ResNet):
    def __init__(self, in_shape, num_classes):
        super(ResNet101, self).__init__(
            Bottleneck, [3, 4, 23, 3], in_shape, num_classes
        )


class ResNet152(ResNet):
    def __init__(self, in_shape, num_classes):
        super(ResNet152, self).__init__(
            Bottleneck, [3, 8, 36, 3], in_shape, num_classes
        )


class ResNetEncoder(ResNet):
    def __init__(self, in_shape, num_classes):
        super(ResNetEncoder, self).__init__(BasicBlock, [2, 2, 2, 2], in_shape, FC1_OUT)
        self.fc2 = FakeLinear(FC1_OUT, FC2_OUT, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["FakeLinear", "Linear", "Conv2d", "BatchNorm2d"],
            register_self=True,
        )

    def set_delta(self, delta, indices):
        self.hook.backward_return = delta
        self.hook.batch_indices = indices

    def forward(self, x):
        x = super(ResNetEncoder, self).forward(x)
        x = self.fc2(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(ResNetDecoder, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, FC3_OUT, bias=False)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(FC3_OUT, n_classes, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "FakeLinear"],
            register_self=True,
        )

    def forward(self, x):
        x = self.fc3(x)
        x = self.relu1(x)
        x = self.fc4(x)
        x = self.relu2(x)
        return x


class ResNetEncoderDecoder(ResNet):
    def __init__(self, input_shape, n_classes):
        super(ResNetEncoderDecoder, self).__init__(
            BasicBlock, [2, 2, 2, 2], input_shape, FC1_OUT
        )
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["FakeLinear", "Linear", "Conv2d", "BatchNorm2d"],
            register_self=True,
        )
        self.fc2 = FakeLinear(FC1_OUT, FC2_OUT, bias=False)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, FC3_OUT, bias=False)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(FC3_OUT, n_classes, bias=False)

    def forward(self, x):
        x = super(ResNetEncoderDecoder, self).forward(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.relu2(x)
        x = self.fc4(x)
        return x
