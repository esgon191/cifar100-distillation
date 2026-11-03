from clearml import Dataset, Logger


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
