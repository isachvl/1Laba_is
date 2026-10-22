"""Trainer отвечает только за обучение и проверку нейросети.

Код почти такой же, как в исходном проекте. Новое здесь — on_epoch:
передаём Optuna текущую точность, чтобы прунер мог прервать плохой trial.
"""

import math
import time

import mlflow
import torch
from torch import nn


class Trainer:
    def __init__(self, model, data, config, results):
        self.model = model
        self.data = data
        self.config = config
        self.results = results

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )

    def train_epoch(self):
        """Одна эпоха: делаем прогноз, считаем loss и обновляем веса."""
        self.model.train()
        train_loss = 0.0

        for images, labels in self.data.train_loader:
            self.optimizer.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            # Учитываем размер батча для правильного среднего loss.
            train_loss += loss.item() * len(labels)

        return train_loss / len(self.data.train_dataset)

    def validate(self):
        """Проверяем качество на val, не меняя веса модели."""
        self.model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in self.data.val_loader:
                output = self.model(images)
                loss = self.criterion(output, labels)

                val_loss += loss.item() * len(labels)
                prediction = torch.argmax(output, dim=1)
                correct += (prediction == labels).sum().item()

        val_loss /= len(self.data.val_dataset)
        val_acc = correct / len(self.data.val_dataset)
        return val_loss, val_acc

    def train(self, on_epoch=None):
        """Все 30 эпох; on_epoch вызывается после каждой эпохи для Optuna."""
        epochs = self.config["train"]["epochs"]

        for epoch in range(1, epochs + 1):
            start = time.perf_counter()

            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            epoch_time = time.perf_counter() - start

            # Если обучение испортилось численно, trial отмечается как FAIL.
            if not all(math.isfinite(x) for x in (train_loss, val_loss, val_acc)):
                return float("nan")

            self.results.save_epoch(
                epoch, train_loss, val_loss, val_acc, epoch_time
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

            print(
                f"Эпоха {epoch}/{epochs} | "
                f"Ошибка обучения: {train_loss:.4f} | "
                f"Ошибка проверки: {val_loss:.4f} | "
                f"Точность: {val_acc * 100:.2f}%"
            )

            if on_epoch is not None:
                on_epoch(epoch, val_acc)

        return val_acc
