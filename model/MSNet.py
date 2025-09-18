from model.down import DownWithMaxAndAvg
import torch
from torch import Tensor
from torch import nn
from model.layer_normal import LayerNorm2d

from torchvision.ops import DeformConv2d
from model.WMSA import ResidualWindowAttention
from einops import rearrange


class SpatialAttention(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DCNGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super(DCNGate, self).__init__()

        self.norm = LayerNorm2d(channels)
        self.conv_in = nn.Conv2d(channels, channels * 2, 1, 1, 0)
        self.conv_out = nn.Conv2d(channels, channels, 1, 1, 0)

        self.docnv1 = DeformConv2d(channels, channels, kernel_size=7, padding=3)
        self.docnv2 = DeformConv2d(channels, channels, kernel_size=5, padding=2)
        self.docnv3 = DeformConv2d(channels, channels, kernel_size=3, padding=1)

        self.offset_conv1 = nn.Conv2d(channels, 2 * 7 * 7, 5, padding=2)
        self.offset_conv2 = nn.Conv2d(channels, 2 * 5 * 5, 5, padding=2)
        self.offset_conv3 = nn.Conv2d(channels, 2 * 3 * 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        tmp = x
        x = self.norm(x)
        x = self.conv_in(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        offset = self.offset_conv1(x1)
        x1 = self.docnv1(x1, offset)
        offset = self.offset_conv2(x1)
        x1 = self.docnv2(x1, offset)
        offset = self.offset_conv3(x1)
        x1 = self.docnv3(x1, offset)
        x = x1 + x2
        x = self.conv_out(x)

        return x + tmp


class DCNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(DCNBlock, self).__init__()

        self.conv_in = nn.Conv2d(channels, channels * 2, 1)
        self.DCN1 = ResidualWindowAttention(channels * 2)
        self.DCN2 = ResidualWindowAttention(channels * 2)
        self.conv_out = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        x = self.DCN1(x)
        x = self.DCN2(x)
        x = self.conv_out(x)
        x = self.act(x)

        return x


class DpConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1, bias=False) -> None:
        super(DpConv2d, self).__init__()

        self.__conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=bias)
        self.__conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                  bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.__conv_1(x)
        x = self.__conv_2(x)
        return x


class ShiftGateUnit(nn.Module):
    def __init__(self, channels: int, shift_dim: int, shift: int, conv_scale: tuple) -> None:
        super(ShiftGateUnit, self).__init__()

        self.__shift_dim = shift_dim
        self.__shift = shift

        self.__conv_1 = nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=conv_scale[0],
                                  stride=1, padding=conv_scale[1], groups=channels // 2, bias=True)
        self.__conv_2 = nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=3, dilation=2,
                                  stride=1, padding=2, groups=channels // 2, bias=True)
        self.__conv_3 = nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=2, dilation=2,
                                  stride=1, padding=1, groups=channels // 2, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        patchs = torch.chunk(x_1, 8, dim=1)
        patchs = list(patchs)

        patchs[0] = torch.roll(patchs[0], shifts=[self.__shift], dims=[self.__shift_dim])
        patchs[2] = torch.roll(patchs[2], shifts=[self.__shift], dims=[self.__shift_dim])
        patchs[4] = torch.roll(patchs[4], shifts=[self.__shift], dims=[self.__shift_dim])
        patchs[6] = torch.roll(patchs[6], shifts=[self.__shift], dims=[self.__shift_dim])

        x_1 = torch.cat(patchs, dim=1)
        x_1 = self.__conv_1(x_1)
        x_1 = self.__conv_2(x_1)
        x_1 = self.__conv_3(x_1)

        return x_1 + x_2


class ShiftSerialUnit(nn.Module):
    def __init__(self, channels: int, shift_dim: int, shift: int, conv_scale: tuple) -> None:
        super(ShiftSerialUnit, self).__init__()

        self.__shift_dim = shift_dim
        self.__shift = shift

        self.__conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=conv_scale[0],
                                  stride=1, padding=conv_scale[1], groups=channels, bias=True)
        self.__conv_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, dilation=2,
                                  stride=1, padding=2, groups=channels, bias=True)
        self.__conv_3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=2, dilation=2,
                                  stride=1, padding=1, groups=channels, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        patchs = torch.chunk(x, 8, dim=1)
        patchs = list(patchs)

        patchs[0] = torch.roll(patchs[0], shifts=[self.__shift], dims=[self.__shift_dim])
        patchs[2] = torch.roll(patchs[2], shifts=[self.__shift], dims=[self.__shift_dim])
        patchs[4] = torch.roll(patchs[4], shifts=[self.__shift], dims=[self.__shift_dim])
        patchs[6] = torch.roll(patchs[6], shifts=[self.__shift], dims=[self.__shift_dim])

        x_1 = torch.cat(patchs, dim=1)
        x_1 = self.__conv_1(x_1)
        x_1 = self.__conv_2(x_1)
        x_1 = self.__conv_3(x_1)

        return x_1


class BCSBlock(nn.Module):
    def __init__(self, channels: int, shift, conv_scale):
        """
        Bidirectional Channel Shift Block
        :param channels:
        """
        super(BCSBlock, self).__init__()

        self.__norm = LayerNorm2d(channels=channels)

        self.__conv_1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.__conv_2 = DpConv2d(in_channels=channels, out_channels=channels * 2)

        # self.__avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.__conv_4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        # self.attn = SpatialAttention(channels)

        self.__conv_5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.__conv_6 = DpConv2d(in_channels=channels, out_channels=channels * 2)

        self.__conv_7 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.__activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

        self.__shift_gate_1 = ShiftGateUnit(channels=channels * 2, shift_dim=2, shift=shift, conv_scale=conv_scale)
        self.__shift_gate_2 = ShiftGateUnit(channels=channels * 2, shift_dim=3, shift=shift, conv_scale=conv_scale)

    def forward(self, x: Tensor) -> Tensor:
        temp = x
        x = self.__norm(x)
        x = self.__conv_1(x)
        x = self.__conv_2(x)

        x = self.__shift_gate_1(x)
        x = self.__activation(x)

        # w = self.__avg(x)
        # w = self.__conv_4(w)
        # x = x * w
        # x = self.attn(x) * x

        x = self.__conv_5(x)
        x = x + temp * self.beta
        temp = x

        x = self.__norm(x)
        x = self.__conv_6(x)
        x = self.__shift_gate_2(x)

        x = self.__conv_7(x)
        x = self.__activation(x)
        return x + temp * self.gamma


class MSNet(nn.Module):
    def __init__(self, num_features: int, encoder_blocks: [], decoder_blocks: [], bottleneck_blocks: int,
                 shift_length: int):
        """
        Multi-scale Shift Network
        :param num_features:
        :param encoder_blocks:
        :param decoder_blocks:
        :param bottleneck_blocks:
        """
        super(MSNet, self).__init__()

        self.__project_in = nn.Conv2d(in_channels=3, out_channels=num_features, kernel_size=3, stride=1, padding=1)
        self.__project_out = nn.Conv2d(in_channels=num_features, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.__encoder_1 = nn.Sequential(
            *[BCSBlock(channels=num_features, shift=shift_length, conv_scale=(7, 3)) for _ in range(encoder_blocks[0])]
            # *[DCNBlock(channels=num_features) for _ in range(encoder_blocks[0])]
        )

        self.__encoder_2 = nn.Sequential(
            *[BCSBlock(channels=num_features * 2, shift=shift_length, conv_scale=(7, 3)) for _ in
              range(encoder_blocks[1])]
            # *[DCNBlock(channels=num_features * 2) for _ in range(encoder_blocks[1])]
        )

        self.__encoder_3 = nn.Sequential(
            *[BCSBlock(channels=num_features * 4, shift=shift_length, conv_scale=(7, 3)) for _ in
              range(encoder_blocks[2])]
            # *[DCNBlock(channels=num_features * 4) for _ in range(encoder_blocks[2])]
        )

        self.__bottleneck = nn.Sequential(
            *[BCSBlock(channels=num_features * 8, shift=shift_length, conv_scale=(7, 3)) for _ in
              range(bottleneck_blocks)]
            # *[DCNBlock(channels=num_features * 8) for _ in range(bottleneck_blocks)]
        )

        self.__decoder_1 = nn.Sequential(
            *[BCSBlock(channels=num_features * 4, shift=shift_length, conv_scale=(7, 3)) for _ in
              range(decoder_blocks[0])]
            # *[DCNBlock(channels=num_features * 4) for _ in range(decoder_blocks[2])]
        )

        self.__decoder_2 = nn.Sequential(
            *[BCSBlock(channels=num_features * 2, shift=shift_length, conv_scale=(7, 3)) for _ in
              range(decoder_blocks[1])]
            # *[DCNBlock(channels=num_features * 2) for _ in range(decoder_blocks[1])]
        )

        self.__decoder_3 = nn.Sequential(
            *[BCSBlock(channels=num_features, shift=shift_length, conv_scale=(7, 3)) for _ in range(decoder_blocks[2])]
            # *[DCNBlock(channels=num_features) for _ in range(decoder_blocks[0])]
        )

        self.__down_1 = DownWithMaxAndAvg(in_channels=num_features, out_channels=num_features * 2)
        self.__down_2 = DownWithMaxAndAvg(in_channels=num_features * 2, out_channels=num_features * 4)
        self.__down_3 = DownWithMaxAndAvg(in_channels=num_features * 4, out_channels=num_features * 8)

        self.__up_1 = nn.PixelShuffle(2)
        self.__up_2 = nn.PixelShuffle(2)
        self.__up_3 = nn.PixelShuffle(2)

    def forward(self, x):
        temp = x

        x = self.__project_in(x)

        x = self.__encoder_1(x)
        x = self.__down_1(x)
        temp_1 = x

        x = self.__encoder_2(x)
        x = self.__down_2(x)
        temp_2 = x

        x = self.__encoder_3(x)
        x = self.__down_3(x)
        temp_3 = x

        x = self.__bottleneck(x)
        x = torch.cat((x, temp_3), dim=1)

        x = self.__up_1(x)
        x = self.__decoder_1(x)
        x = torch.cat((x, temp_2), dim=1)

        x = self.__up_2(x)
        x = self.__decoder_2(x)
        x = torch.cat((x, temp_1), dim=1)

        x = self.__up_3(x)
        x = self.__decoder_3(x)

        x = self.__project_out(x)
        return x + temp
