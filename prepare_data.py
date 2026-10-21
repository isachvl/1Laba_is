from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import json
import numpy as np
from pathlib import Path
import hashlib
import sklearn

X,y = load_digits(return_X_y=True)

idx = np.arange(len(y))

rest, test = train_test_split(
    idx,
    test_size=0.2, #20% test
    stratify=y, #stratify - это параметр, который позволяет сохранять примерно одинаковое соотношение классов при разделении.
    random_state=0
)
 
train, val = train_test_split(
    rest,
    test_size=0.25,#25% для проверки из текстовых
    stratify=y[rest],
    random_state=0
)

 
print("Всего:", len(X))
print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
 
assert len(set(train) & set(val)) == 0
assert len(set(train) & set(test)) == 0
assert len(set(val) & set(test)) == 0

 
assert len(train) + len(val) + len(test) == 1797

for part in [train, val, test]:
    assert len(np.unique(y[part])) == 10


# Сохраняем индексы в JSON
split = {
    "train": train.tolist(),
    "val": val.tolist(),
    "test": test.tolist()
}

path = "data/split.json"

with open(path, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)

print(f"Разбиение сохранено: {path}")

 
split_path = Path("data/split.json")

sha256 = hashlib.sha256(
    split_path.read_bytes()
).hexdigest()

manifest = {
    "source": "sklearn.datasets.load_digits",
    "version": sklearn.__version__,
    "total_records": len(y),
    "split_seed": 0,
    "split_rule": "Stratified: test=20%, val=25% of rest",
    "train_records": len(train),
    "val_records": len(val),
    "test_records": len(test),
    "excluded_records": 0,
    "split_file": "data/split.json",
    "split_sha256": sha256
}

with open("data/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

pathSHA = "data/manifest.json"

print("SHA-256:", sha256)
print(f"Манифест сохранён: {pathSHA}")
