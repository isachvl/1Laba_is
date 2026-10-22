# ЛР0 — дополнения к репозиторию isachvl/1Laba_is

Файлы подготовлены **на основе твоего проекта**. Названия классов и методы сохранены:
`DigitMLP` → `DigitsData` → `Trainer` → `Results` → `Experiment`.

**Что не трогаем:** `model.py`, `data.py`, `prepare_data.py`, `configs/base.yaml`, `data/split.json`, `data/manifest.json`, `requirements.txt`, `hypothesis.md`, `runs/`, `mlflow.db`.

**Что обновляется:** `train.py`, `trainer.py`, `results.py`, `experiment.py`, `.gitignore`.

**Что добавляется:** `optimize.py`, `run_final.py`, `make_report.py`, `verify.py`, `run_all.ps1`, `configs/search.yaml`, `hypothesis_example.md`, `requirements_lr0.txt`.

## Как установить (PowerShell)

1. Сначала убедись, что твои локальные изменения сохранены в Git (`git status`). Старый код можно вернуть с помощью Git.
2. Распакуй содержимое архива **поверх** папки `C:\Users\Влэд\Desktop\1laba`, с заменой одноимённых файлов. Не удаляй проект и его папку `runs`.
3. Активируй окружение и установи нужные библиотеки:

```powershell
cd C:\Users\Влэд\Desktop\1laba
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements_lr0.txt
python verify.py
```

`verify.py` **не перезаписывает** `data/split.json` и **не читает тестовые предсказания**.

## Проведение экспериментов

В репозитории файл `hypothesis.md` на момент просмотра оказался пустым. Проверь его:
перед **Optuna** запиши туда прогноз по улучшению на val и по переносу на
новые seed/test. Прогноз исходного исследования seed=1–5 задним числом не приписывай.
Шаблон — `hypothesis_example.md`.

**Обязательно сначала закоммить код + гипотезу:**

```powershell
git add .
git commit -m "Prepare Optuna search for LR0"
git push
```

После этого весь пайплайн можно выполнить одной командой:

```powershell
.\run_all.ps1
```

Этот скрипт последовательно:
1. Проверяет данные, метрики и настройки.
2. Запускает **10** испытаний Optuna (записывает их в `optuna.db`).
3. Возобновляет ту же study **до 40**, не переобучая предыдущие.
4. Создаёт `configs/best.yaml` и фиксирует его локальным Git-коммитом.
5. Выполняет **10 финальных запусков**: `base` и `best` с `seed=101...105`. Test используется **только здесь**.
6. Готовит таблицы, графики и Word-отчёт.

Или запускай по частям:

```powershell
python optimize.py --target 10
python optimize.py --target 40
# после второго шага: git add configs/best.yaml ; git commit -m "Lock best config"
python run_final.py
python make_report.py
```

### Что получится

- `runs/<run_id>/`: конфигурация, метаданные, CSV по эпохам и изображениям, итоговые метрики;
- `optuna.db`: 40 записей, в том числе COMPLETE / PRUNED / FAIL;
- `mlflow.db` и `mlartifacts/`: параметры, метрики по эпохам, артефакты;
- `configs/best.yaml`: лучшая конфигурация после 40 trial;
- `report/LR0_report.docx`: черновик отчёта на основе **реальных** результатов;
- `report/*.csv`, `report/optuna_accuracy.png`, `report/optuna_importance.png`.

Проверь созданный отчёт перед сдачей. Не придумывай результаты для ещё не проведённых экспериментов.

### MLflow

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db
```

Открой http://127.0.0.1:5000 и выбери **Model training**.

### Методические замечания

`history.csv` содержит `epoch_time`, поэтому два запуска с одинаковым seed
совпадают по ошибкам и точности, но **не обязаны совпадать побайтно по времени**.
В отчёте это важно указать честно.

После прерывания `run_all.ps1` безопасно отдельно повторить
`python optimize.py --target 40`. У TPE после перезапуска не гарантируется
та же последовательность предложений, что при непрерывном запуске.

Завершив все шаги, сохрани исходники, `configs/best.yaml` и `report/` в Git:

``` терминал
git add .
git commit -m "Finish LR0 with final results and report"
git push
```

Каталоги запусков и SQLite-базы по `.gitignore` не попадут на GitHub: их нужно
сдать отдельно или выгрузить CSV через `make_report.py`.
