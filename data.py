import json
import torch

from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader


class DigitsData:

    def __init__(self, batch_size, generator):
 
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
        self.val_dataset = TensorDataset(X_val, y_val)

        # Загрузчик обучающих данных
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
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