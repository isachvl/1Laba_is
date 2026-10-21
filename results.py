import csv
import json
from pathlib import Path
import csv
import json
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
    def save_predictions(
        self,
        model,
        val_loader,
        val_idx,
        test_loader=None,
        test_idx=None
    ):

        model.eval()

        with open(self.predictions_path, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["split", "index", "y_true", "y_pred"])

            # Проверочная выборка
            self.write_predictions(
                model, val_loader, val_idx, "val", writer
            )

            # Тестовая выборка, если разрешена
            if test_loader is not None:
                self.write_predictions(
                    model, test_loader, test_idx, "test", writer
                )

        print("Предсказания сохранены:", self.predictions_path)
    def save_metadata(self, metadata):

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False
            )
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
    def write_predictions(self, model, loader, indices, split_name, writer):

        offset = 0

        with torch.no_grad():

            for images, labels in loader:

                output = model(images)

                predictions = torch.argmax(output, dim=1)

                for j in range(len(labels)):

                    writer.writerow([
                        split_name,
                        indices[offset + j],
                        labels[j].item(),
                        predictions[j].item()
                    ])

                offset += len(labels)
    # Сохраняем итоговые метрики
    def save_metrics(self, expected_val_acc=None):

        # Читаем сохранённые предсказания
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            predictions = list(csv.DictReader(f))

        # Читаем историю обучения
        with open(self.history_path, "r", encoding="utf-8") as f:
            history = list(csv.DictReader(f))

        # Отбираем проверочную выборку
        val_rows = [
            row for row in predictions
            if row["split"] == "val"
        ]

        # Вычисляем точность из CSV
        val_correct = sum(
            int(row["y_true"]) == int(row["y_pred"])
            for row in val_rows
        )

        metrics = {
            "val_acc": val_correct / len(val_rows),
            "val_loss": float(history[-1]["val_loss"]),
            "train_time": sum(
                float(row["epoch_time"]) for row in history
            )
        }

        # Отбираем тестовую выборку
        test_rows = [
            row for row in predictions
            if row["split"] == "test"
        ]

        # Если тестовые предсказания есть
        if test_rows:

            test_correct = sum(
                int(row["y_true"]) == int(row["y_pred"])
                for row in test_rows
            )

            metrics["test_acc"] = test_correct / len(test_rows)

        # Сохраняем итоговые метрики
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print("Итоговые метрики:", metrics)

        return metrics