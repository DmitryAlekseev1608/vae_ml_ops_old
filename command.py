import fire
from app.train import train
from app.infer import infer

def start_train():
  train()

def start_infer():
  infer()

if __name__ == '__main__':
  fire.Fire()
