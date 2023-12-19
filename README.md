export PATH="/home/oem/miniconda3/bin:$PATH" conda create --prefix
/home/oem/Desktop/vae_ml_ops/conda_env_gen python=3.9 --no-default-packages conda activate
"/home/dmitry/Рабочий стол/ClassificationPointNet3D/miniconda_venv"

export PATH="~/.pyenv/bin:$PATH"

pip install poetry poetry install --no-root Запустите для возможности работы pre-commit
install isort, black, flake8, pylint, prettier Проверка без коммита pre-commit run
--all-files

dvc pull --remote gd_vae

python3 com.py +args_cli=train python3 com.py +args_cli=infer

mlflow server --host 127.0.0.1 --port 8080
