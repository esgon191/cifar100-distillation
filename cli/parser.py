import argparse

# Все известные параметры по умолчанию
# Набор будет расширяться по мере реализации шагов (модели, KD-лосс)
DEFAULTS = {
    "project_name": "cifar100-distillation",
    "output_dir": "out",
    "val_size": 5000,
    "split_seed": 42,
    "seed": 0,
    "learning_rate": 0.1,
    "per_device_train_batch_size": 128,
    "per_device_eval_batch_size": 512,
    "num_train_epochs": 200,
    "weight_decay": 5e-4,
    "num_labels": 100,
}

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="training",
        description="CLI для обучения учителя и учеников на CIFAR-100"
    )

    p.add_argument(
        "--project_name",
        type=str,
        default=DEFAULTS["project_name"],
        help="Проект в ClearML"
    )

    p.add_argument(
        "--dataset_name",
        type=str,
        help="Название датасета в ClearML"
    )

    p.add_argument(
        "--dataset_version",
        type=str,
        help="Версия датасета в ClearML"
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULTS["output_dir"],
        help="Путь для сохранения модели и логов"
    )

    p.add_argument(
        "--val_size",
        type=int,
        default=DEFAULTS["val_size"],
        help="Число примеров train, отводимых под валидацию"
    )

    p.add_argument(
        "--split_seed",
        type=int,
        default=DEFAULTS["split_seed"],
        help="Сид разбиения train/val (не зависит от сида обучения)"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Сид обучения (инициализация, порядок батчей, аугментации)"
    )

    p.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULTS["learning_rate"],
        help="Скорость обучения"
    )

    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULTS["per_device_train_batch_size"],
        help="Батч на устройство для обучения"
    )

    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=DEFAULTS["per_device_eval_batch_size"],
        help="Батч на устройство для валидации"
    )

    p.add_argument(
        "--num_train_epochs",
        type=int,
        default=DEFAULTS["num_train_epochs"],
        help="Число эпох"
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULTS["weight_decay"],
        help="L2 регуляризация"
    )

    p.add_argument(
        "--num_labels",
        type=int,
        default=DEFAULTS["num_labels"],
        help="Число классов целевой задачи"
    )

    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Включить 16-битное обучение (AMP, GPU only)"
    )

    return p
