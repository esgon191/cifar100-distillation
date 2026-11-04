import argparse

from kd.models import ARCHS

# Все известные параметры по умолчанию
# Набор будет расширяться по мере реализации шагов (KD-лосс)
DEFAULTS = {
    "project_name": "cifar100-distillation",
    "data_dir": "data/cifar_100",
    "output_dir": "out",
    "val_size": 5000,
    "split_seed": 42,
    "seed": 0,
    "arch": "resnet50",
    "cnn_width": 32,
    "cnn_convs_per_stage": 2,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "warmup_epochs": 1,
    "label_smoothing": 0.0,
    "per_device_train_batch_size": 128,
    "per_device_eval_batch_size": 512,
    "num_train_epochs": 200,
    "weight_decay": 5e-4,
    "weight_decay_norm_bias": True,
    "num_labels": 100,
    "eval_every": 1,
    "save_every": 5,
}

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="training",
        description="CLI для обучения учителя и учеников на CIFAR-100"
    )

    # --- ClearML ---

    p.add_argument(
        "--project_name",
        type=str,
        default=DEFAULTS["project_name"],
        help="Проект в ClearML"
    )

    p.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Имя задачи в ClearML (по умолчанию <arch>-seed<seed>)"
    )

    p.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=[],
        help="Теги задачи ClearML (по ним собираются результаты для статистики)"
    )

    p.add_argument(
        "--clearml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Логировать запуск в ClearML (--no-clearml — для локальных smoke-тестов)"
    )

    p.add_argument(
        "--continue_task_id",
        type=str,
        default=None,
        help="ID задачи ClearML, которую продолжить (после обрыва сессии Colab)"
    )

    # --- Данные ---

    p.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULTS["data_dir"],
        help="Папка с CIFAR-100 (pickle train/test/meta); если пусто — скачивается"
    )

    p.add_argument(
        "--dataset_name",
        type=str,
        help="Название датасета в ClearML (если задано — вместо --data_dir)"
    )

    p.add_argument(
        "--dataset_version",
        type=str,
        help="Версия датасета в ClearML"
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
        "--max_train_samples",
        type=int,
        default=None,
        help="Ограничить train первыми N примерами (для отладки)"
    )

    # --- Модель ---

    p.add_argument(
        "--arch",
        type=str,
        choices=ARCHS,
        default=DEFAULTS["arch"],
        help="Архитектура: resnet50 (учитель) или cnn (ученик)"
    )

    p.add_argument(
        "--cnn_width",
        type=int,
        default=DEFAULTS["cnn_width"],
        help="Ширина первой стадии CNN-ученика (далее ×2, ×4)"
    )

    p.add_argument(
        "--cnn_convs_per_stage",
        type=int,
        default=DEFAULTS["cnn_convs_per_stage"],
        help="Число conv-блоков в каждой из 3 стадий CNN-ученика"
    )

    p.add_argument(
        "--num_labels",
        type=int,
        default=DEFAULTS["num_labels"],
        help="Число классов целевой задачи"
    )

    # --- Обучение ---

    p.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULTS["output_dir"],
        help="Путь для сохранения модели и чекпоинтов (в Colab — на Google Drive)"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Сид обучения (инициализация, порядок батчей, аугментации)"
    )

    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Детерминированные алгоритмы cuDNN (медленнее)"
    )

    p.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULTS["learning_rate"],
        help="Скорость обучения"
    )

    p.add_argument(
        "--momentum",
        type=float,
        default=DEFAULTS["momentum"],
        help="Momentum SGD"
    )

    p.add_argument(
        "--nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nesterov momentum"
    )

    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=DEFAULTS["warmup_epochs"],
        help="Эпох линейного warmup перед косинусным спадом LR"
    )

    p.add_argument(
        "--label_smoothing",
        type=float,
        default=DEFAULTS["label_smoothing"],
        help="Label smoothing для CE"
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
        "--weight_decay_norm_bias",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["weight_decay_norm_bias"],
        help="Применять weight decay к BatchNorm и bias "
             "(--no-weight_decay_norm_bias — только к весам conv/linear)"
    )

    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Включить 16-битное обучение (AMP, GPU only)"
    )

    p.add_argument(
        "--eval_every",
        type=int,
        default=DEFAULTS["eval_every"],
        help="Оценивать на val каждые N эпох"
    )

    # --- Чекпоинты ---

    p.add_argument(
        "--save_every",
        type=int,
        default=DEFAULTS["save_every"],
        help="Сохранять чекпоинт каждые N эпох"
    )

    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Продолжить с чекпоинта этого запуска, если он есть"
    )

    p.add_argument(
        "--upload_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Дублировать чекпоинт артефактом ClearML (перезаписывается, один на задачу)"
    )

    return p
