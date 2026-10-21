import random
import numpy as np
import torch
import yaml

from pathlib import Path
from datetime import datetime
from uuid import uuid4
from model import DigitMLP
from data import DigitsData
from trainer import Trainer
from results import Results
import sys
import hashlib
import subprocess

from importlib.metadata import version

class Experiment:

    def __init__(self, config_path, seed=None, eval_test=False):

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if seed is not None:
            self.config["seed"] = seed

        # Сохраняем параметр тестирования в объекте
        self.eval_test = eval_test

        self.validate_config()

    # Фиксируем случайность
    def set_seed(self):

        seed = self.config["seed"]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.use_deterministic_algorithms(True)

        torch.set_num_threads(
            self.config["runtime"]["num_threads"]
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        return generator

    # Запускаем эксперимент
    def run(self):

        # Фиксируем случайность
        generator = self.set_seed()

        # Создаём нейросеть
        model = DigitMLP(**self.config["model"])

        # Загружаем данные
        data = DigitsData(
            batch_size=self.config["train"]["batch_size"],
            generator=generator,
            eval_test=self.eval_test
        )

        print("Обучающих изображений:", len(data.train_dataset))
        print("Проверочных изображений:", len(data.val_dataset))

        # Создаём уникальный идентификатор запуска
        run_id = (
            f"baseline_seed{self.config['seed']}_"
            f"{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid4().hex[:8]}"
        )
    
        # Путь к папке запуска
        run_dir = Path("runs") / run_id

        # Создаём объект сохранения результатов
        results = Results(run_dir)

        # Сохраняем настройки текущего эксперимента
        config_path = run_dir / "config.yaml"

        metadata = self.create_metadata(run_id)

        results.save_metadata(metadata)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.config,
                f,
                allow_unicode=True,
                sort_keys=False
            )

        print("Идентификатор запуска:", run_id)

        trainer = Trainer(
            model=model,
            data=data,
            config=self.config,
            results=results
        )

        try:

            val_acc = trainer.train()

            results.save_predictions(
                model,
                data.val_loader,
                data.val_idx
            )

            results.save_metrics(val_acc)

            metadata["status"] = "complete"

        except Exception as error:

            metadata["status"] = "failed"
            metadata["error"] = str(error)

            raise

        finally:

            metadata["end_time"] = datetime.now().astimezone().isoformat()

            results.save_metadata(metadata)
    def create_metadata(self, run_id):

        # Получаем SHA256 разделения данных
        split_path = Path("data/split.json")

        split_hash = hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest()

        # Получаем текущий Git-коммит
        try:
            git_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True
            )

            git_commit = (
                git_result.stdout.strip()
                if git_result.returncode == 0
                else None
            )

        except FileNotFoundError:
            git_commit = None

        # Формируем словарь метаданных
        metadata = {
            "run_id": run_id,
            "series": "baseline",
            "status": "running",
            "seed": self.config["seed"],
            "split_seed": self.config["split_seed"],
            "start_time": datetime.now().astimezone().isoformat(),
            "end_time": None,
            "git_commit": git_commit,
            "split_sha256": split_hash,
            "num_threads": torch.get_num_threads(),

            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "sklearn": version("scikit-learn"),
                "optuna": version("optuna"),
                "mlflow": version("mlflow")
            }
        }

        return metadata
    def validate_config(self):

        model = self.config["model"]
        train = self.config["train"]
        runtime = self.config["runtime"]

        # Проверка скорости обучения
        if train["lr"] <= 0:
            raise ValueError("lr должен быть больше 0")

        # Проверка количества эпох
        if train["epochs"] <= 0:
            raise ValueError("epochs должен быть больше 0")

        # Проверка размера батча
        if train["batch_size"] <= 0:
            raise ValueError("batch_size должен быть больше 0")

        # Проверка числа скрытых слоёв
        if model["n_layers"] < 1:
            raise ValueError("n_layers должен быть не меньше 1")

        # Проверка количества нейронов
        if model["hidden"] <= 0:
            raise ValueError("hidden должен быть больше 0")

        # Проверка Dropout
        if not 0 <= model["dropout"] < 1:
            raise ValueError("dropout должен быть от 0 до 1 (не включая 1)")

        # Проверка регуляризации
        if train["weight_decay"] < 0:
            raise ValueError("weight_decay не может быть отрицательным")

        # Проверка количества потоков
        if runtime["num_threads"] < 1:
            raise ValueError("num_threads должен быть не меньше 1")