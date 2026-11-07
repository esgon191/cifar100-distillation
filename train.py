"""
Обучение модели на CIFAR-100 (учитель ResNet-50 или CNN-ученик) с логированием
в ClearML. Пример:
    python train.py --arch resnet50 --seed 0 --tags teacher
    python train.py --arch cnn --cnn_width 32 --n_runs 10 --init_seed 0 --tags student-ce

--n_runs запусков серии отличаются только начальными весами (сиды из --init_seed);
порядок батчей, аугментации и прочую случайность задаёт общий --seed
(см. kd/experiment.py). --parallel_runs запусков обучаются одновременно одной группой.

Все результаты пишутся на диск в <output_dir>/<task_id>/ (в Colab — Google Drive);
во время обучения в ClearML уходит только консоль. В конце серии таблицы
собираются из файлов, проверяется, что все файлы на месте, и в ClearML
отправляются кривые обучения, single values и таблицы-артефакты
(--report_only — только этот последний шаг)
"""
import json
import os

from cli.parser import get_parser
from cli.validators import validate_args
from kd.data import DeviceLoader, load_cifar100, stratified_split
from kd.experiment import (
    CrossEntropyExperiment,
    Loaders,
    aggregate_final,
    collect_series,
    verify_series_files,
)
from kd.models import model_spec
from kd.trainer import get_device, set_seed
from utils.clearml_utils import get_clearml_dataset, report_single_values


def default_task_name(args) -> str:
    """Имя задачи по умолчанию: архитектура, ёмкость, сиды и число запусков"""
    arch = args.arch if args.arch != "cnn" else f"cnn-w{args.cnn_width}-c{args.cnn_convs_per_stage}"
    name = f"{arch}-seed{args.seed}-init{args.init_seed}"
    return name if args.n_runs == 1 else f"{name}-x{args.n_runs}"


def init_task(args):
    """
    Создаёт (или продолжает, если задан --continue_task_id) задачу ClearML.
    Артефакты и модели — на files server ClearML: адрес задаётся явно, т.к.
    output_uri=True подхватил бы default_output_uri (S3) из ~/clearml.conf.
    Автологирование torch.save отключено, чтобы промежуточные чекпоинты
    не расходовали хранилище.

    Продолжение: каталог запуска — out/<task_id>, поэтому id продолжаемой
    задачи обязан совпасть с переданным (иначе чекпоинт не найдётся и обучение
    молча начнётся заново). Без --continue_task_id всегда создаётся новая
    задача (reuse_last_task_id=False), иначе ClearML мог бы переиспользовать
    недавний «пустой» запуск с тем же именем
    """
    from clearml import Task
    from clearml.backend_api.session import Session

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.training,
        tags=[args.arch, *args.tags],
        output_uri=Session.get_files_server_host(),
        reuse_last_task_id=False,
        continue_last_task=args.continue_task_id or False,
        # TF/TensorBoard не используем; в Colab их биндинг лишь импортирует TensorFlow
        auto_connect_frameworks={"pytorch": False, "tensorflow": False, "tensorboard": False},
    )
    # На воркере (clearml-agent) id задачи задаёт очередь — проверка только для ручного запуска
    if args.continue_task_id and Task.running_locally() and task.id != args.continue_task_id:
        raise RuntimeError(
            f"Не удалось продолжить задачу {args.continue_task_id} "
            f"(ClearML создал новую: {task.id}) — проверь id"
        )
    # Метрики репортятся с явным номером эпохи; при продолжении ClearML сдвигает
    # итерации на последнюю отправленную — отключаем сдвиг, чтобы графики не «уезжали»
    task.set_initial_iteration(0)
    task.connect(args, name="Args")
    return task


def report_series(task, args, run_dir: str, tables: dict) -> None:
    """
    Итог серии в ClearML — только из файлов на диске (collect_series):
    - кривые обучения: график = метрика, series = train/val, значение —
      среднее по запускам на эпоху (при n_runs=1 — сама кривая запуска);
    - single values финального замера: среднее и std по запускам;
    - артефакты-таблицы test_metrics (главный: метрики финальных моделей на test,
      строка = запуск), final_metrics, history и series (параметры серии);
    - веса первого запуска (остальные — в <run_dir>/run_<i>/)
    """
    from clearml import OutputModel

    logger = task.get_logger()
    curves = tables["history"].groupby(["metric", "split", "epoch"])["value"].mean()
    for (metric, split, epoch), value in curves.items():
        logger.report_scalar(title=metric, series=split, value=float(value), iteration=int(epoch))

    for split, values in aggregate_final(tables["final_metrics"]).items():
        report_single_values(split, values)
    with open(os.path.join(run_dir, "series.json")) as f:
        series = json.load(f)
    report_single_values("model", {"n_params": series["n_params"], "n_runs": series["n_runs"]})

    for name in ("test_metrics", "final_metrics", "history"):
        task.upload_artifact(name, artifact_object=tables[name])
    task.upload_artifact("series", artifact_object=series)
    # Только веса первого запуска — хранилище files server ограничено
    OutputModel(task=task, name=f"{args.task_name}-run0", framework="PyTorch").update_weights(
        weights_filename=os.path.join(run_dir, "run_0", "model_final.pt"), auto_delete_file=False
    )


def main():
    args = get_parser().parse_args()
    validate_args(args)
    if args.task_name is None:
        args.task_name = default_task_name(args)

    task = init_task(args) if args.clearml else None
    # Абсолютный путь: все файлы серии — только здесь (в Colab — на Google Drive)
    run_dir = os.path.abspath(
        os.path.join(args.output_dir, task.id if task is not None else args.task_name)
    )
    print(f"Каталог серии: {run_dir}")

    if not args.report_only:
        train_series(args, task, run_dir)

    tables = collect_series(run_dir, args.n_runs)
    verify_series_files(run_dir, args.n_runs)
    test = tables["test_metrics"]["top1"]
    std = f" ± {test.std(ddof=1):.4f}" if len(test) > 1 else ""
    print(f"Серия {args.task_name}: {len(test)} запусков, test top1 {test.mean():.4f}{std}; "
          f"файлы — {run_dir}")

    if task is not None:
        report_series(task, args, run_dir, tables)
        task.close()


def train_series(args, task, run_dir: str) -> None:
    """Данные, загрузчики и обучение серии запусков (все результаты — в run_dir)"""
    # Глобальный сид: порядок батчей, аугментации и прочая случайность (кроме весов)
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    channels_last = device.type == "cuda"
    print(f"Устройство: {device}, AMP: {args.fp16 and device.type == 'cuda'}")

    # --- Данные ---
    data_dir = args.data_dir
    if args.dataset_name:
        data_dir = get_clearml_dataset(args.dataset_name, args.dataset_version, args.project_name)
    data = load_cifar100(data_dir)

    train_idx, val_idx = stratified_split(data.y_train, args.val_size, args.split_seed)
    if args.max_train_samples is not None:
        train_idx = train_idx[:args.max_train_samples]
    print(f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(data.y_test)}")

    eval_kwargs = dict(
        batch_size=args.per_device_eval_batch_size, device=device,
        channels_last=channels_last
    )
    loaders = Loaders(
        train=DeviceLoader(
            data.x_train[train_idx], data.y_train[train_idx],
            batch_size=args.per_device_train_batch_size, device=device,
            shuffle=True, augment=True, drop_last=True, seed=args.seed,
            channels_last=channels_last
        ),
        val=DeviceLoader(data.x_train[val_idx], data.y_train[val_idx], **eval_kwargs),
        test=DeviceLoader(data.x_test, data.y_test, **eval_kwargs),
        # Те же примеры train без аугментаций — для финального замера
        train_eval=DeviceLoader(data.x_train[train_idx], data.y_train[train_idx], **eval_kwargs),
    )

    # --- Серия запусков ---
    model_cls, model_kwargs = model_spec(args)
    os.makedirs(run_dir, exist_ok=True)
    experiment = CrossEntropyExperiment(
        args, model_cls, model_kwargs, loaders, device, run_dir, task=task
    )
    print(f"{args.task_name}: запусков {args.n_runs}, init_seed {args.init_seed}, "
          f"одновременно {args.parallel_runs}")
    experiment.run(args.n_runs, args.init_seed)


if __name__ == "__main__":
    main()
