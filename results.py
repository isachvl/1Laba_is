import csv
import json
from pathlib import Path

import torch


class Results:

    def __init__(self, run_dir):

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.run_dir / "history.csv"
        self.predictions_path = self.run_dir / "predictions.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        self.metadata_path = self.run_dir / "metadata.json"

        # Создаём history.csv с заголовками
        with open(self.history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "epoch_time"
            ])

    # Сохраняем результат одной эпохи
    def save_epoch(self, epoch, train_loss, val_loss, val_acc, epoch_time):

        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_acc,
                epoch_time
            ])

    # Сохраняем предсказания нейросети
    def save_predictions(self, model, val_loader, val_idx):

        model.eval()

        with open(self.predictions_path, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["split", "index", "y_true", "y_pred"])

            offset = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    output = model(images)
                    predictions = torch.argmax(output, dim=1)

                    for j in range(len(labels)):

                        writer.writerow([
                            "val",
                            val_idx[offset + j],
                            labels[j].item(),
                            predictions[j].item()
                        ])

                    offset += len(labels)

        print("Предсказания сохранены:", self.predictions_path)
    def save_metadata(self, metadata):

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False
            )
    # Сохраняем итоговые метрики
    def save_metrics(self, expected_val_acc):

        # Читаем сохранённые предсказания
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        correct = sum(
            int(row["y_true"]) == int(row["y_pred"])
            for row in rows
        )

        accuracy = correct / len(rows)

        # Проверяем совпадение точности
        assert abs(accuracy - expected_val_acc) < 1e-12

        # Читаем историю обучения
        with open(self.history_path, "r", encoding="utf-8") as f:
            history = list(csv.DictReader(f))

        val_loss = float(history[-1]["val_loss"])

        train_time = sum(
            float(row["epoch_time"])
            for row in history
        )

        metrics = {
            "val_acc": accuracy,
            "val_loss": val_loss,
            "train_time": train_time
        }

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print("Правильных ответов:", correct)
        print(f"Точность из CSV: {accuracy * 100:.2f}%")
        print("Итоговые метрики:", metrics)