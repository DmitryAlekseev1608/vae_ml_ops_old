import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

from .dataset import fetch_dataset
from .loss_func import loss_vae
from .model import LinearVAE


def train():
    """Функция обучения модели"""

    all_photos, all_attrs = fetch_dataset()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    criterion = loss_vae

    autoencoder = LinearVAE().to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters())

    train_photos, val_photos, _, _ = train_test_split(
        all_photos, all_attrs, train_size=0.9, shuffle=False
    )
    train_loader = torch.utils.data.DataLoader(train_photos, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val_photos, batch_size=32)

    n_epochs = 5
    train_losses = []
    val_losses = []

    for _ in tqdm.tqdm(range(n_epochs), ncols=100):

        autoencoder.train()
        train_losses_per_epoch = []
        for batch in train_loader:
            optimizer.zero_grad()
            reconstruction, mu, logsigma = autoencoder(batch.to(device))
            reconstruction = reconstruction.view(-1, 64, 64, 3)
            loss = criterion(batch.to(device).float(), mu, logsigma, reconstruction)
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())

        train_losses.append(np.mean(train_losses_per_epoch))

        autoencoder.eval()
        val_losses_per_epoch = []
        with torch.no_grad():
            for batch in val_loader:
                reconstruction, mu, logsigma = autoencoder(batch.to(device))
                reconstruction = reconstruction.view(-1, 64, 64, 3)
                loss = criterion(batch.to(device).float(), mu, logsigma, reconstruction)
                val_losses_per_epoch.append(loss.item())

        val_losses.append(np.mean(val_losses_per_epoch))

    torch.save(autoencoder.state_dict(), "models/autoencoder")
