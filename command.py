import fire

from app.infer import infer
from app.train import train


def start_train():
    train()


def start_infer():
    infer()


if __name__ == '__main__':
    fire.Fire()
