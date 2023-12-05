export PATH="/home/USERNAME/miniconda3/bin:$PATH" conda create --prefix
/home/oem/Desktop/vae_ml_ops/conda_env_gen python=3 --no-default-packages conda activate
"/home/dmitry/Рабочий стол/ClassificationPointNet3D/miniconda_venv"

pip install poetry poetry install --no-root

python3 command.py start_train python3 command.py start_infer

Запустите для возможности работы pre-commit install

isort, black, flake8, pylint, prettier Проверка без коммита pre-commit run --all-files
