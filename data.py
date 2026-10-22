import json
import torch

from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader
"""загружаем данные и преоразуем датаЛодырей"""
class DigitsData:

    def __init__(self, batch_size, generator, eval_test=False):
 
        X, y = load_digits(return_X_y=True)

        # Загружаем сохранённое разделение
        with open("data/split.json", "r", encoding="utf-8") as f:
            split = json.load(f)

        train_idx = split["train"]
        self.val_idx = split["val"]

        # Обучающие данные
        X_train = torch.tensor(
            X[train_idx] / 16.0,#нормализуем
            dtype=torch.float32
        )

        y_train = torch.tensor(
            y[train_idx],
            dtype=torch.long
        )

        # Проверочные данные
        X_val = torch.tensor(
            X[self.val_idx] / 16.0,
            dtype=torch.float32
        )

        y_val = torch.tensor(
            y[self.val_idx],
            dtype=torch.long
        )

        # Создаём датасеты
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val) # объедниение правильного изображение и ответа

        # Загрузчик обучающих данных
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,#перемешивать изображения перед каждой эпохой.
            num_workers=0,
            generator=generator
        )

        # Загрузчик проверочных данных
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
                # По умолчанию тестовые данные не используются
        self.test_loader = None
        self.test_idx = None

        # Загружаем test только при --eval-test
        if eval_test:

            self.test_idx = split["test"]

            X_test = torch.tensor(
                X[self.test_idx] / 16.0,
                dtype=torch.float32
            )

            y_test = torch.tensor(
                y[self.test_idx],
                dtype=torch.long
            )

            self.test_dataset = TensorDataset(X_test, y_test)

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )