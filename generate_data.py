import os

import hydra
from omegaconf import DictConfig, OmegaConf

from neuroment2.mixing import Mixer


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    OmegaConf.to_container()
    kwargs = OmegaConf.to_container(cfg["Mixer"])  # we only want a raw dict, DictConfig types may be trouble
    a = Mixer(**kwargs)


if __name__ == "__main__":
    main()
