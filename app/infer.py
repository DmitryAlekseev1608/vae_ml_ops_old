import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from .model import LinearVAE

def infer():

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  autoencoder = LinearVAE()
  autoencoder.load_state_dict(torch.load("models/autoencoder"))
  autoencoder.eval()

  z = np.array([np.random.normal(0, 1, 16) for i in range(10)])
  output = autoencoder.sample(torch.FloatTensor(z).to(device))

  plt.figure(figsize=(18, 18))
  for i in range(output.shape[0]):
    plt.subplot(output.shape[0] // 2, 2, i + 1)
    generated = output[i].cpu().detach().numpy()
    plt.imshow(generated)

  plt.show()