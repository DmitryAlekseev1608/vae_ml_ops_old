export PATH="/home/oem/miniconda3/bin:$PATH"
conda create --prefix /home/oem/Desktop/vae_ml_ops/conda_env_gen python=3.9 --no-default-packages 
conda activate "/home/oem/Desktop/vae_ml_ops/conda_env_gen"
pip install poetry
poetry env use /home/oem/Desktop/vae_ml_ops/conda_env_gen/bin/python3
poetry config virtualenvs.path /home/oem/Desktop/vae_ml_ops/conda_env_gen
poetry install --no-root
Запустите для возможности работы
pre-commit install
isort, black, flake8, pylint, prettier
Проверка без коммита
pre-commit run --all-files
dvc pull --remote gd_vae
mlflow server --host 127.0.0.1 --port 8080
python3 commands.py train
python3 commands.py infer

python3 commands.py train