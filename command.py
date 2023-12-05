import fire

from src.infer import infer
from src.train import train


def start_train():
    """
    Function to start model's train
    """
    train()


def start_infer():
    """
    Function to start model's infer
    """
    infer()


if __name__ == '__main__':
    fire.Fire()
