from src.infer import infer
from src.train import train
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(cfg: DictConfig):

    if cfg.args_cli == "train":
        train(cfg)

    if cfg.args_cli == "infer":
        infer(cfg)

if __name__ == '__main__':
    start()