"""Один запуск обучения. Все параметры можно передавать через терминал."""

import argparse
from pathlib import Path

from experiment import Experiment


def main():
    parser = argparse.ArgumentParser(
        description="ЛР0: обучение нейросети для распознавания цифр"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--eval-test", action="store_true",
        help="Использовать test только в окончательном сравнении",
    )
    args = parser.parse_args()

    # Для обычного запуска test НЕ используется.
    series = "final" if args.eval_test else "baseline"
    variant = Path(args.config).stem if args.eval_test else None

    experiment = Experiment(
        config_path=args.config,
        seed=args.seed,
        eval_test=args.eval_test,
        series=series,
        variant=variant,
    )
    experiment.run()


if __name__ == "__main__":
    main()
