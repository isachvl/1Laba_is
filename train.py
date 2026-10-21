import argparse
from experiment import Experiment


def main():

    parser = argparse.ArgumentParser(
        description="Обучение нейросети для распознавания цифр"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Путь к файлу конфигурации"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Начальное значение генератора случайных чисел"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Оценить модель на тестовой выборке"
    )
    args = parser.parse_args()

    experiment = Experiment(
        config_path=args.config,
        seed=args.seed,
        eval_test=args.eval_test
    )

    experiment.run()


if __name__ == "__main__":
    main()