import hydra
import logging as log
from omegaconf import DictConfig
import torch
from torch import nn
from torchinfo import summary


class NeuromentModel(nn.Module):
    def __init__(
        self,
        num_input_features=1025,
        num_instruments=34,  # our number of classes
        use_batch_norm=True,
    ):
        super(NeuromentModel, self).__init__()

        self.num_input_features = num_input_features
        self.num_instruments = num_instruments
        self.use_batch_norm = use_batch_norm

        # if self.use_batch_norm:
        #     self.fc_in = nn.Sequential(
        #         nn.Linear(self.num_input_features, 128),
        #         SwapDim((0, 2, 1)),
        #         nn.BatchNorm1d(num_features=128),
        #         SwapDim((0, 2, 1)),
        #         nn.ReLU(),
        #     )
        # else:
        #     self.fc_in = nn.Sequential(
        #         nn.Linear(self.n_input_features, 128),
        #         nn.ReLU(),
        #     )

        self.conv_1 = ConvBlock(1, 16, (3, 3), stride=(1, 1), padding=(3 // 2, 3 // 2))
        self.conv_2 = ConvBlock(16, 32, (3, 3), stride=(1, 1), padding=(3 // 2, 3 // 2))

        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_3 = ConvBlock(32, 48, (5, 5), stride=(1, 1), padding=(5 // 2, 5 // 2))
        self.conv_4 = ConvBlock(48, 64, (5, 5), stride=(1, 1), padding=(5 // 2, 5 // 2))

        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_5 = ConvBlock(64, 80, (7, 7), stride=(1, 1), padding=(7 // 2, 7 // 2))
        self.conv_6 = ConvBlock(80, 96, (7, 7), stride=(1, 1), padding=(7 // 2, 7 // 2))

        self.flatten = nn.Flatten()

        self.out = nn.Sequential(
            nn.Linear(int(96 * self.num_input_features / 4 * 100 / 4), out_features=10,),
            nn.Sigmoid(),
        )

        #
        # self.conv_1 = nn.Conv2d(
        #     1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        # )64
        # self.bn_1 = nn.BatchNorm2d(num_features=16)
        # self.act_1 = nn.ReLU()
        #
        # self.conv_2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        # self.bn_2 = nn.BatchNorm2d(num_features=32)
        # self.act_2 = nn.ReLU()

        summary_str = summary(
            self, input_size=(1, 1, self.num_input_features, 100), verbose=0
        )
        log.info(summary_str)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.pool_1(x)

        x = self.conv_3(x)
        x = self.conv_4(x)

        x = self.pool_2(x)

        x = self.conv_5(x)
        x = self.conv_6(x)

        x = self.flatten(x)

        x = self.out(x)




class ConvBlock(nn.Module):
    """Packs Conv2d + BatchNorm + ReLU activation into a module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class SwapDim(nn.Module):
    """Swaps dimensions of a tensor."""

    def __init__(self, swap_order=()):
        super(SwapDim, self).__init__()
        self.swap_order = swap_order

    def forward(self, x):
        return torch.permute(x, self.swap_order)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = NeuromentModel()


if __name__ == "__main__":
    main()
