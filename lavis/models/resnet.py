
import torch.nn.functional as F
import torch
from torch import nn
from lavis.models import weight_init
import torchvision


class LightweightConv(nn.Module):
    def __init__(self, out_dim=768) -> None:
        super(LightweightConv, self).__init__()

        self.stem = BasicStem(
            in_channels=3,
            out_channels=64,
        )

        self.block1 = BasicBlock(in_channels=64, out_channels=128, stride=2)
        self.block2 = BasicBlock(in_channels=128, out_channels=256, stride=2)
        self.block3 = BasicBlock(in_channels=256, out_channels=512, stride=1)

        self.up_dim = nn.Conv2d(in_channels=512, out_channels=out_dim, kernel_size=1)

    def forward(self, x):
        y1 = self.stem(x)
        y2 = self.block1(y1)
        y3 = self.block2(y2)
        y4 = self.block3(y3)
        out = self.up_dim(y4)
        return out

class Res18Wrapper(nn.Module):
    def __init__(self, out_channels) -> None:
        super(Res18Wrapper, self).__init__()
        self.conv_res18 = torchvision.models.resnet18(pretrained=True)
        self.out_project = nn.Conv2d(in_channels=512, out_channels=out_channels,
                                     kernel_size=1)

        del self.conv_res18.avgpool
        del self.conv_res18.fc

    def forward(self, x):
        x = self.conv_res18.conv1(x)
        x = self.conv_res18.bn1(x)
        x = self.conv_res18.relu(x)
        x = self.conv_res18.maxpool(x)

        x = self.conv_res18.layer1(x)
        x = self.conv_res18.layer2(x)
        x = self.conv_res18.layer3(x)
        x = self.conv_res18.layer4(x)

        x = self.out_project(x)

        conv_out = F.interpolate(
            x, scale_factor=2, mode="nearest") # 16x reduce

        return conv_out



class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=nn.BatchNorm2d(out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x

    @property
    def out_channels(self):
        if self.deep_stem:
            return self.conv1_3.out_channels
        else:
            return self.conv1.out_channels

    @property
    def stride(self):
        return 4  # = stride 2 conv -> stride 2 max pool



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1):
        """
        The standard block type for ResNet18 and ResNet34.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): A callable that takes the number of
                channels and returns a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.activation = nn.ReLU()

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=nn.BatchNorm2d(out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.BatchNorm2d(out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = self.activation(out)
        return out



class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



