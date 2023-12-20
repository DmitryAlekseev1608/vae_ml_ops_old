from os.path import dirname, split

import hydra
import mlflow
import torch
from git import Repo
from mlflow import MlflowClient
from omegaconf import DictConfig

from vae_ml_ops.infer import infer
from vae_ml_ops.train import train


def _get_git_commit_id():

    """Return the ID of the git HEAD commit."""

    path = split(dirname(__file__))[0] + "/vae_ml_ops"
    commit_id = Repo(path).head.object.hexsha

    return commit_id[:8]


def print_auto_logged_info(r):
    "Print logs"
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def start(cfg: DictConfig):

    """Point entry"""

    if cfg.args_cli == "train":

        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("train VAE")

        with mlflow.start_run() as run:

            commit_id = _get_git_commit_id()
            params = {
                "Epoch": cfg.train.n_epochs,
                "Image_size": cfg.model.size_img,
                "Features": cfg.model.features,
                "commit_id": commit_id,
            }

            mlflow.log_params(params)

            autoencoder = train(cfg)

            mlflow.set_tag("Training Info", "VAE model for fetch_dataset")

            mlflow.pytorch.log_model(autoencoder, "model")
            scripted_pytorch_model = torch.jit.script(autoencoder)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

            for artifact_path in ["model/data", "scripted_model/data"]:
                [
                    f.path
                    for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
                ]

        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    if cfg.args_cli == "infer":
        infer(cfg)


if __name__ == '__main__':
    start()
