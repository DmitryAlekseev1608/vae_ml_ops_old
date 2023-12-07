import fire
from hydra import compose, initialize

from src.infer import infer
from src.train import train


def start_train():
    """
    Function to start model's train
    """
    train(cfg)


def start_infer():
    """
    Function to start model's infer
    """
    infer(cfg)


if __name__ == '__main__':

    initialize(version_base=None, config_path="conf", job_name="app")
    cfg = compose(config_name="config", return_hydra_config=True)

    fire.Fire()
