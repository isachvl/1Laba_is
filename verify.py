"""Проверки каркаса без переобучения и без использования test."""

import csv
import hashlib
import json
from pathlib import Path

from sklearn.datasets import load_digits

from experiment import Experiment


def load_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def verify_split():
    """Проверяем исходный split.json; НЕ создаём его заново."""
    path = Path("data/split.json")
    split = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    train, val, test = (set(split[name]) for name in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (1077, 360, 360)
    assert not (train & val or train & test or val & test)
    assert len(train | val | test) == 1797
    digits = load_digits()
    for part in (train, val, test):
        assert len(set(digits.target[list(part)])) == 10
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = manifest.get("split_sha256", manifest.get("sha256_split_json"))
    assert actual_hash == expected_hash, "Не совпадает хеш разделения!"
    print("Данные, индексы и SHA-256: OK")


def verify_config():
    experiment = Experiment("configs/base.yaml")
    old_value = experiment.config["train"]["lr"]
    experiment.config["train"]["lr"] = -1
    try:
        experiment.validate_config()
        raise AssertionError("Отрицательная скорость обучения была принята")
    except ValueError:
        pass
    finally:
        experiment.config["train"]["lr"] = old_value
    print("Проверка отрицательного lr: OK")


def verify_metrics():
    count = 0
    for path in Path("runs").glob("*/metrics.json"):
        pred_path = path.parent / "predictions.csv"
        if not pred_path.exists():
            continue
        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        rows = [r for r in load_csv(pred_path) if r["split"] == "val"]
        if not rows:
            continue
        correct = sum(r["y_true"] == r["y_pred"] for r in rows)
        assert abs(correct / len(rows) - metrics["val_acc"]) < 1e-10
        count += 1
    print(f"Точность из CSV совпала с metrics.json: {count} запусков")


def verify_reproducibility():
    """Сравниваем ошибки и ответы; реальное время выполнения не обязано совпасть."""
    runs = {}
    for path in Path("runs").glob("*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        seed = metadata.get("seed")
        if (metadata.get("status") not in ("complete", "completed")
                or metadata.get("series") not in ("baseline", "base_seeds")
                or seed not in (0, 1)):
            continue
        if (path.parent / "predictions.csv").exists():
            runs.setdefault(seed, []).append(path.parent)

    if len(runs.get(0, [])) >= 2:
        first, second = sorted(runs[0])[-2:]
        a = load_csv(first / "history.csv")
        b = load_csv(second / "history.csv")
        metric_names = ("epoch", "train_loss", "val_loss", "val_acc")
        assert [[r[k] for k in metric_names] for r in a] == [
            [r[k] for k in metric_names] for r in b
        ]
        assert (first / "predictions.csv").read_bytes() == (
            second / "predictions.csv"
        ).read_bytes()
        print("Два запуска seed=0: метрики и предсказания совпали")
    else:
        print("Для сравнения seed=0 нужны два завершённых запуска")

    if runs.get(0) and runs.get(1):
        a = sorted(runs[0])[-1] / "predictions.csv"
        b = sorted(runs[1])[-1] / "predictions.csv"
        assert a.read_bytes() != b.read_bytes()
        print("Seed 0 и 1 дают разные предсказания")


def main():
    assert sum(a == b for a, b in zip([1, 2, 3, 3], [1, 2, 3, 4])) / 4 == 0.75
    print("Тест accuracy=0.75: OK")
    verify_split()
    verify_config()
    verify_metrics()
    verify_reproducibility()
    print("Проверки выполнены. Test не использовался.")


if __name__ == "__main__":
    main()
