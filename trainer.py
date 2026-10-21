import time
import torch

from torch import nn


class Trainer:

    def __init__(self, model, data, config, results):

        self.model = model
        self.data = data
        self.config = config
        self.results = results

        # Функция ошибки
        self.criterion = nn.CrossEntropyLoss()

        # Оптимизатор Adam
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )

    # Обучение одной эпохи
    def train_epoch(self):

        self.model.train()
        train_loss = 0

        for images, labels in self.data.train_loader:

            self.optimizer.zero_grad()

            output = self.model(images)

            loss = self.criterion(output, labels)

            loss.backward()

            self.optimizer.step()

            train_loss += loss.item() * len(labels)

        return train_loss / len(self.data.train_dataset)

    # Проверка нейросети
    def validate(self):

        self.model.eval()

        val_loss = 0
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

    # Полный цикл обучения
    def train(self):

        epochs = self.config["train"]["epochs"]

        for epoch in range(epochs):

            start_time = time.perf_counter()

            # Обучаем одну эпоху
            train_loss = self.train_epoch()

            # Проверяем точность
            val_loss, val_acc = self.validate()

            epoch_time = time.perf_counter() - start_time

            # Сохраняем результаты
            self.results.save_epoch(
                epoch + 1,
                train_loss,
                val_loss,
                val_acc,
                epoch_time
            )

            print(
                f"Эпоха {epoch + 1}/{epochs} | "
                f"Ошибка обучения: {train_loss:.4f} | "
                f"Ошибка проверки: {val_loss:.4f} | "
                f"Точность: {val_acc * 100:.2f}%"
            )

        return val_acc