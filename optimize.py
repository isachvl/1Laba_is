"""Optuna: 10 испытаний для проверки продолжения, затем до 40.

Запуск:
    python optimize.py --target 10
    python optimize.py --target 40

Каждое испытание вызывает тот же класс Experiment, что и обычный train.py.
"""

import argparse
from pathlib import Path

import mlflow
import optuna
import yaml

from experiment import Experiment, NonFiniteMetricError, get_git_info


STUDY_NAME = "lr0_search"
STUDY_STORAGE = "sqlite:///optuna.db"


def create_study():
    """Открываем существующий поиск или создаём новый."""
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_STORAGE,
        load_if_exists=True,
        direction="maximize",  # Чем выше точность, тем лучше.
        sampler=optuna.samplers.TPESampler(seed=42),# алгос который выбирает какой сеед выбрать
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
    )


def suggest_parameters(trial):
    """Optuna выбирает гиперпараметры в границах задания."""
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "hidden": trial.suggest_int("hidden", 32, 512, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def create_trial_config(params):
    """Берём твою базовую YAML и меняем только шесть параметров."""
    config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    config["seed"] = 0
    config["train"]["epochs"] = 30

    for name in ("n_layers", "hidden", "dropout"):
        config["model"][name] = params[name]
    for name in ("lr", "weight_decay", "batch_size"):
        config["train"][name] = params[name]

    return config


def objective(trial):
    """Обучаем один вариант и возвращаем точность на val после 30 эпох."""
    params = suggest_parameters(trial)
    config = create_trial_config(params)

    # Experiment принимает путь к YAML, поэтому создаём временный файл.
    temp_path = Path("runs") / f"trial_{trial.number}_config.yaml"
    temp_path.parent.mkdir(exist_ok=True)
    temp_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    try:
        result = Experiment(
            config_path=temp_path,
            series="search",
            trial=trial,
        ).run()
        trial.set_user_attr("run_dir", result["run_dir"])
        return result["val_acc"]

    except NonFiniteMetricError:
        # Возврат NaN → Optuna пометит этот trial как FAIL и продолжит поиск.
        return float("nan")

    finally:
        temp_path.unlink(missing_ok=True)


def save_best_config(study):
    """После всех 40 trial фиксируем лучшую конфигурацию без test."""
    best_trial = study.best_trial
    config = create_trial_config(best_trial.params)
    Path("configs/best.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Лучшая попытка: #{best_trial.number}; val_acc={best_trial.value:.2%}")
    print("Конфигурация сохранена: configs/best.yaml")


def main():
    parser = argparse.ArgumentParser(description="Поиск параметров Optuna")
    parser.add_argument(
        "--target", type=int, default=40,
        help="Общее число испытаний, а не число новых (10 или 40)",
    )
    args = parser.parse_args()
    if args.target not in (10, 40):
        parser.error(" --target 10, затем --target 40")

    # Задание требует записать прогноз и зафиксировать код ДО поиска.
    hypothesis = Path("hypothesis.md")
    if not hypothesis.exists() or not hypothesis.read_text(encoding="utf-8").strip():
        raise SystemExit("Заполни hypothesis.md и закоммить его ДО Optuna")
    try:
        _, changes = get_git_info()
    except (RuntimeError, OSError) as error:
        raise SystemExit("Для поиска нужен Git-репозиторий: " + str(error))
    if changes:
        raise SystemExit("Сначала закоммить изменения:\n" + changes)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0")
    study = create_study()

    # Если Python закрыли во время trial, старую незавершённую запись
    # помечаем FAIL, чтобы она не висела в статусе RUNNING.
    for old in study.trials:
        if old.state == optuna.trial.TrialState.RUNNING:
            study.tell(old.number, state=optuna.trial.TrialState.FAIL)
            print(f"Прерванный trial #{old.number} помечен FAIL")

    already_done = len(study.trials)
    if args.target == 40 and already_done == 0:
        raise SystemExit("Сначала проверь продолжение: python optimize.py --target 10")
    if already_done > args.target:
        raise SystemExit(f"В базе уже {already_done} испытаний")

    missing = args.target - already_done
    print(f"Уже есть {already_done}, осталось {missing} до {args.target}")

    if missing > 0:
        with mlflow.start_run(run_name=f"lr0_search_{already_done}_to_{args.target}"):
            mlflow.set_tag("series", "optuna_search")
            mlflow.log_param("sampler_seed", 42)
            mlflow.log_param("target", args.target)
            # n_jobs=1 важен для воспроизводимости.
            study.optimize(objective, n_trials=missing, n_jobs=1)

    print("Всего испытаний:", len(study.trials))
    if len(study.trials) == 40:
        save_best_config(study)


if __name__ == "__main__":
    main()
