import hydra
import logging as log
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torchinfo import summary


class NeuromentModel(nn.Module):
    def __init__(
        self,
        num_instruments,  # our number of classes
        num_input_features,
        num_input_frames,
        use_batch_norm=True,
    ):
        super(NeuromentModel, self).__init__()

        self.num_instruments = num_instruments  # i.e. num_output_features
        self.num_input_features = num_input_features
        self.num_input_frames = num_input_frames
        self.use_batch_norm = use_batch_norm

        pool_size_1 = 2
        pool_size_2 = 2

        self.conv_1 = ConvBlock(1, 16, (3, 3), stride=(1, 1), padding=(3 // 2, 3 // 2))
        self.conv_2 = ConvBlock(16, 32, (3, 3), stride=(1, 1), padding=(3 // 2, 3 // 2))

        self.pool_1 = nn.MaxPool2d(kernel_size=(pool_size_1, 1))

        self.conv_3 = ConvBlock(32, 48, (5, 5), stride=(1, 1), padding=(5 // 2, 5 // 2))
        self.conv_4 = ConvBlock(48, 64, (5, 5), stride=(1, 1), padding=(5 // 2, 5 // 2))

        self.pool_2 = nn.MaxPool2d(kernel_size=(pool_size_2, 1))

        self.conv_5 = ConvBlock(64, 80, (7, 7), stride=(1, 1), padding=(7 // 2, 7 // 2))
        self.conv_6 = ConvBlock(80, 96, (7, 7), stride=(1, 1), padding=(7 // 2, 7 // 2))

        self.flatten = nn.Flatten()

        self.out = nn.Sequential(
            nn.Linear(in_features=int(96 * self.num_input_frames * self.num_input_features
                                      // (pool_size_1 * pool_size_2)),
                      out_features=int(self.num_instruments * self.num_input_frames),),
            nn.Sigmoid(),
            Reshape((-1, self.num_instruments, self.num_input_frames))
        )

        summary_str = summary(
            self,
            input_size=(
                1,
                self.num_input_features,
                self.num_input_frames,
            ),
            verbose=0,
            batch_dim=0,
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

        # x = self.swap_dim(x)
        x = self.flatten(x)

        x = self.out(x)

        return x


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

    def __init__(self, swap_dim_1, swap_dim_2):
        super(SwapDim, self).__init__()
        self.swap_dim_1 = swap_dim_1
        self.swap_dim_2 = swap_dim_2

    def forward(self, x: torch.Tensor):
        return torch.transpose(x, self.swap_dim_1, self.swap_dim_2)


class Reshape(nn.Module):
    """Reshapes a 1D-tensor to a 2D-tensor."""

    def __init__(self, output_shape):
        super(Reshape, self).__init__()
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        return x.view(self.output_shape)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    num_instruments = 7
    num_input_features = 128
    num_input_frames = 14
    batch_size = 16

    model = NeuromentModel(
        num_instruments=num_instruments,
        num_input_features=num_input_features,  # for Mel bins
        num_input_frames=num_input_frames,
        use_batch_norm=True,
    )

    input_tensor = torch.Tensor(
        np.random.normal(size=[batch_size, 1, num_input_features, num_input_frames])
    )
    log.info("Input shape: %s [batch_size, n_channels, n_features, n_frames]" % str(input_tensor.shape))

    output_tensor = model(input_tensor)
    log.info("Output shape: %s [batch_size, n_classes, n_frames]" % str(output_tensor.shape))


if __name__ == "__main__":
    main()
