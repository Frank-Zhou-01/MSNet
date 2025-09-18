from torch import Tensor
from torch import nn


class DownWithAvgPool(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownWithAvgPool, self).__init__()

        self.__conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.__avg = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.__conv(x)
        x = self.__avg(x)

        return x


class DownWithMaxPool(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownWithMaxPool, self).__init__()

        self.__conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.__max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.__conv(x)
        x = self.__max(x)

        return x


class DownWithMaxAndAvg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownWithMaxAndAvg, self).__init__()

        self.__conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.__max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.__avg = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.__conv(x)
        x = self.__max(x) + self.__avg(x)

        return x
