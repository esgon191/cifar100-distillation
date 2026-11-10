from clearml import Dataset, Logger, Task


def download_task_model(task_id: str) -> str:
    """
    Локальный путь к последней выходной модели задачи (для учителя — model_final.pt).
    Файл скачивается с files server в локальный кэш ClearML; повторный вызов
    берёт его оттуда
    """
    models = Task.get_task(task_id=task_id).models["output"]
    if not models:
        raise RuntimeError(f"У задачи ClearML {task_id} нет выходных моделей")
    path = models[-1].get_local_copy()
    if not path:
        raise RuntimeError(f"Не удалось скачать модель задачи {task_id} ({models[-1].url})")
    return path


def get_clearml_dataset(
        dataset_name : str,
        dataset_version : str,
        dataset_project : str = 'cifar100-distillation'
) -> str:
    """
    Скачивает версию ClearML Dataset и возвращает путь
    к локальной (read-only) копии
    """
    ds = Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_version=dataset_version
    )

    # Путь к папке с датасетом
    path = ds.get_local_copy()

    return path


def report_metrics(
        title: str,
        metrics: dict,
        iteration: int
) -> None:
    """
    Репорт словаря метрик в ClearML в виде скаляров
    (title — группа на вкладке Scalars, ключи — series)
    """
    logger = Logger.current_logger()
    if logger is None:
        print('logger недоступен и равен None')
        return

    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.report_scalar(
                title=title,
                series=metric,
                value=value,
                iteration=iteration
            )


def report_split_metrics(
        split: str,
        metrics: dict,
        iteration: int
) -> None:
    """
    Репорт метрик одной части данных (train/val): ключ — график на вкладке
    Scalars, split — series. Так train и val одной метрики (лосс, CE, KL, top1)
    оказываются на одном графике
    """
    logger = Logger.current_logger()
    if logger is None:
        print('logger недоступен и равен None')
        return

    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.report_scalar(
                title=metric,
                series=split,
                value=value,
                iteration=iteration
            )


def report_single_values(
        prefix: str,
        metrics: dict
) -> None:
    """
    Итоговые метрики как single values (видны в таблице экспериментов
    и доступны при агрегации по множеству задач)
    """
    logger = Logger.current_logger()
    if logger is None:
        print('logger недоступен и равен None')
        return

    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.report_single_value(name=f"{prefix}/{metric}", value=value)
