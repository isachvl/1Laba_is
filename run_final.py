"""Финальное сравнение: base и best при seed от 101 до 105.

До этого скрипта test не используется. Повторный запуск пропускает
уже законченные пары при том же SHA Git-коммита.
"""

import json
from pathlib import Path

from experiment import Experiment, check_final_allowed


def is_finished(variant, seed, commit):
    """Ищем существующий финальный запуск, чтобы не повторять его."""
    for path in Path("runs").glob("*/metadata.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        same_experiment = (
            metadata.get("status") in ("complete", "completed")
            and metadata.get("series") == "final"
            and metadata.get("variant") == variant
            and metadata.get("seed") == seed
            and metadata.get("git_commit") == commit
            and metadata.get("eval_test") is True
        )
        if same_experiment and (path.parent / "metrics.json").exists():
            return True

    return False


def main():
    commit = check_final_allowed()

    for seed in range(101, 106):
        for variant in ("base", "best"):
            if is_finished(variant, seed, commit):
                print(f"Уже выполнено: {variant}, seed={seed}")
                continue

            Experiment(
                config_path=f"configs/{variant}.yaml",
                seed=seed,
                eval_test=True,
                series="final",
                variant=variant,
            ).run()

    print("Финальные 10 запусков готовы. Теперь: python make_report.py")


if __name__ == "__main__":
    main()
