"""Results сохраняет файлы запуска и считает точность только из CSV."""

import csv
import json
from pathlib import Path

import torch


class Results:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.history_path = self.run_dir / "history.csv"
        self.predictions_path = self.run_dir / "predictions.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        self.metadata_path = self.run_dir / "metadata.json"

        # Заголовки CSV создаём один раз, строки эпох дописываем позже.
        with self.history_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "epoch", "train_loss", "val_loss", "val_acc", "epoch_time"
            ])

    def save_epoch(self, epoch, train_loss, val_loss, val_acc, epoch_time):
        """Дописываем метрики одной эпохи в history.csv."""
        with self.history_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                epoch, train_loss, val_loss, val_acc, epoch_time
            ])

    def save_predictions(
        self, model, val_loader, val_idx, test_loader=None, test_idx=None
    ):
        """Всегда сохраняем val, а test — только для финальной оценки."""
        model.eval()
        with self.predictions_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["split", "index", "y_true", "y_pred"])

            self.write_predictions(model, val_loader, val_idx, "val", writer)
            if test_loader is not None:
                self.write_predictions(model, test_loader, test_idx, "test", writer)

    def write_predictions(self, model, loader, indices, split_name, writer):
        """Записываем предсказания, сохраняя исходный индекс изображения."""
        offset = 0

        with torch.no_grad():
            for images, labels in loader:
                predictions = torch.argmax(model(images), dim=1)

                for j in range(len(labels)):
                    writer.writerow([
                        split_name,
                        int(indices[offset + j]),
                        int(labels[j]),
                        int(predictions[j]),
                    ])

                offset += len(labels)

    def save_metrics(self, expected_val_acc=None):
        """Пересчитываем итоговую точность по файлам, а не по переменной."""
        with self.predictions_path.open(encoding="utf-8", newline="") as file:
            predictions = list(csv.DictReader(file))

        with self.history_path.open(encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
        if not history:
            raise ValueError("Не записано ни одной эпохи")

        metrics = {}
        for split_name in ("val", "test"):
            rows = [row for row in predictions if row["split"] == split_name]
            if not rows:
                continue

            correct = sum(
                int(row["y_true"]) == int(row["y_pred"])
                for row in rows
            )
            metrics[f"{split_name}_acc"] = correct / len(rows)

        metrics["val_loss"] = float(history[-1]["val_loss"])
        metrics["train_time"] = sum(
            float(row["epoch_time"]) for row in history
        )

        if expected_val_acc is not None:
            if abs(metrics["val_acc"] - expected_val_acc) > 1e-10:
                raise AssertionError("val_acc не совпадает с predictions.csv")

        self.metrics_path.write_text(
            json.dumps(metrics, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        return metrics

    def save_metadata(self, metadata):
        self.metadata_path.write_text(
            json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8"
        )
