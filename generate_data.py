import os

import hydra
from omegaconf import DictConfig

from neuroment2.mixing import Mix, Mixer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    kwargs = cfg["Mixer"]
    a = Mixer(**kwargs)


if __name__ == "__main__":
    main()
