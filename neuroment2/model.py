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

        summary_str = summary(self, input_size=(1, 100, self.num_input_features), verbose=0)
        log.info(summary_str)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model = NeuromentModel()


class SwapDim(nn.Module):
    """ Swaps dimensions of a tensor.
    """
    def __init__(self, swap_order=()):
        super(SwapDim, self).__init__()
        self.swap_order = swap_order

    def forward(self, x):
        return torch.permute(x, self.swap_order)


if __name__ == "__main__":
    main()
