"""Один запуск эксперимента: настройки обучение файлы MLflow.

Класс Experiment используют train.py, optimize.py и run_final.py.
"""

import hashlib
import math
import random
import subprocess
import sys
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import mlflow
import numpy as np
import optuna
import torch
import yaml

from data import DigitsData
from model import DigitMLP
from results import Results
from trainer import Trainer


class NonFiniteMetricError(Exception):
    """Ошибка для trial, где получилась NaN/бесконечность."""


def get_git_info():
    """Возвращает SHA текущего коммита и список незакоммиченных изменений."""
    def git(*arguments):
        process = subprocess.run(
            ["git", *arguments], capture_output=True, text=True
        )
        if process.returncode != 0:
            raise RuntimeError("Git: " + process.stderr.strip())
        return process.stdout.strip()

    return git("rev-parse", "HEAD"), git("status", "--porcelain")


def current_time():
    return datetime.now().astimezone().isoformat()


def library_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def check_final_allowed():
    """Test разрешён только после 40 trial и при чистом Git."""
    if not Path("configs/best.yaml").exists():
        raise RuntimeError("Сначала заверши Optuna и создай configs/best.yaml")
    if not Path("optuna.db").exists():
        raise RuntimeError("Нет optuna.db — сначала выполни поиск гиперпараметров")

    study = optuna.load_study(
        study_name="lr0_search", storage="sqlite:///optuna.db"
    )
    if len(study.trials) != 40:
        raise RuntimeError(f"Нужно 40 испытаний Optuna, найдено {len(study.trials)}")

    commit, changes = get_git_info()
    if changes:
        raise RuntimeError(
            "Перед финальным тестом сохрани код и best.yaml в Git:\n" + changes
        )
    return commit


class Experiment:
    def __init__(
        self, config_path="configs/base.yaml", seed=None, eval_test=False,
        series="baseline", variant=None, trial=None
    ):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

        if seed is not None:
            self.config["seed"] = seed

        self.eval_test = eval_test
        self.series = series
        self.variant = variant
        self.trial = trial

        self.validate_config()

        # Подбор параметров не должен использовать тестовую выборку.
        if self.eval_test:
            if self.series != "final" or self.variant not in ("base", "best"):
                raise ValueError("--eval-test доступен только для финальной серии")
            if self.config["seed"] not in range(101, 106):
                raise ValueError("В финале используем только seed от 101 до 105")
            expected_path = Path(f"configs/{self.variant}.yaml").resolve()
            if self.config_path.resolve() != expected_path:
                raise ValueError("В финале можно использовать только base.yaml или best.yaml")

    def validate_config(self):
        """Проверяем параметры до обучения, а не после появления ошибки Adam."""
        config = self.config
        model = config["model"]
        train = config["train"]
        runtime = config["runtime"]

        if config["experiment"] != "lr0" or config["split_seed"] != 0:
            raise ValueError("Для ЛР0 experiment=lr0, split_seed=0")
        if type(config["seed"]) is not int or config["seed"] < 0:
            raise ValueError("seed должен быть целым неотрицательным числом")
        if type(model["n_layers"]) is not int or not 1 <= model["n_layers"] <= 3:
            raise ValueError("n_layers должен быть от 1 до 3")
        if type(model["hidden"]) is not int or model["hidden"] <= 0:
            raise ValueError("hidden должен быть положительным целым числом")
        if not 0 <= model["dropout"] < 1:
            raise ValueError("dropout должен быть от 0 до 1 (не включая 1)")
        if not math.isfinite(train["lr"]) or train["lr"] <= 0:
            raise ValueError("lr должен быть больше 0")
        if not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
            raise ValueError("weight_decay не может быть отрицательным")
        if type(train["batch_size"]) is not int or train["batch_size"] < 1:
            raise ValueError("batch_size должен быть положительным целым числом")
        if type(train["epochs"]) is not int or train["epochs"] != 30:
            raise ValueError("По условию ЛР0 должно быть ровно 30 эпох")
        if runtime["device"] != "cpu":
            raise ValueError("Для воспроизводимости в ЛР0 используем CPU")
        if type(runtime["num_threads"]) is not int or runtime["num_threads"] < 1:
            raise ValueError("num_threads должен быть >= 1")

    def set_seed(self):
        """Фиксируем веса, dropout и перемешивание обучающих изображений."""
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.config["runtime"]["num_threads"])

        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def create_metadata(self, run_id, commit):
        split_bytes = Path("data/split.json").read_bytes()
        return {
            "run_id": run_id,
            "series": self.series,
            "variant": self.variant,
            "status": "running",
            "seed": self.config["seed"],
            "split_seed": self.config["split_seed"],
            "trial_number": self.trial.number if self.trial is not None else None,
            "eval_test": self.eval_test,
            "start_time": current_time(),
            "end_time": None,
            "git_commit": commit,
            "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "num_threads": torch.get_num_threads(),
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "sklearn": library_version("scikit-learn"),
                "optuna": library_version("optuna"),
                "mlflow": library_version("mlflow"),
            },
        }

    def _save_config(self, run_dir):
        """В каждом запуске лежит именно та конфигурация, что использовалась."""
        path = run_dir / "config.yaml"
        path.write_text(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _log_params(self, commit, run_id):
        """Параметры и теги для просмотра экспериментов в MLflow."""
        mlflow.set_tags({
            "commit": commit or "uncommitted",
            "series": self.series,
            "seed": str(self.config["seed"]),
            "variant": self.variant or "",
            "run_id": run_id,
        })
        mlflow.log_params({
            "seed": self.config["seed"],
            "split_seed": self.config["split_seed"],
        })
        for section in ("model", "train", "runtime"):
            params = self.config[section]
            for name, value in params.items():
                mlflow.log_param(f"{section}.{name}", value)

        if self.trial is not None:
            mlflow.set_tag("trial_number", str(self.trial.number))

    def run(self):
        """Запускает обучение и возвращает метрики вместе с путём к результатам."""
        if self.eval_test:
            commit = check_final_allowed()
        else:
            try:
                commit, _ = get_git_info()
            except (OSError, RuntimeError):
                commit = None

        generator = self.set_seed()
        model = DigitMLP(**self.config["model"])
        data = DigitsData(
            batch_size=self.config["train"]["batch_size"],
            generator=generator,
            eval_test=self.eval_test,
        )

        # У trial и финальных запусков разные понятные имена.
        name = f"{self.series}_seed{self.config['seed']}"
        if self.variant:
            name += f"_{self.variant}"
        if self.trial is not None:
            name = f"search_trial{self.trial.number:03d}_seed0"
        run_id = f"{name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        run_dir = Path("runs") / run_id

        results = Results(run_dir)
        self._save_config(run_dir)
        metadata = self.create_metadata(run_id, commit)
        results.save_metadata(metadata)

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("lr0")
        print(f"Запуск: {run_id}")
        print(f"Данные: train={len(data.train_dataset)}, val={len(data.val_dataset)}")

        trainer = Trainer(model, data, self.config, results)
        # При поиске Optuna этот запуск будет дочерним для общего MLflow run.
        parent_exists = mlflow.active_run() is not None
        with mlflow.start_run(run_name=run_id, nested=parent_exists):
            self._log_params(commit, run_id)

            try:
                def report_to_optuna(epoch, val_acc):
                    if self.trial is None:
                        return
                    self.trial.report(val_acc, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned(f"Остановлено на эпохе {epoch}")

                val_acc = trainer.train(on_epoch=report_to_optuna)
                if not math.isfinite(val_acc):
                    raise NonFiniteMetricError("val_acc не является конечным числом")

                results.save_predictions(
                    model, data.val_loader, data.val_idx,
                    test_loader=data.test_loader, test_idx=data.test_idx,
                )
                metrics = results.save_metrics(expected_val_acc=val_acc)
                mlflow.log_metrics(metrics, step=self.config["train"]["epochs"])
                metadata["status"] = "completed"
                mlflow.set_tag("state", "COMPLETE")
                print("Итоговые метрики:", metrics)
                return {**metrics, "run_id": run_id, "run_dir": str(run_dir)}

            except optuna.TrialPruned:
                metadata["status"] = "pruned"
                mlflow.set_tag("state", "PRUNED")
                raise

            except Exception as error:
                metadata["status"] = "failed"
                metadata["error"] = str(error)
                metadata["traceback"] = traceback.format_exc()
                mlflow.set_tag("state", "FAIL")
                raise

            finally:
                metadata["end_time"] = current_time()
                results.save_metadata(metadata)
                mlflow.log_artifacts(str(run_dir))
