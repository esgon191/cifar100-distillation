"""
Обучение модели на CIFAR-100 (учитель ResNet-50 или CNN-ученик) с логированием
в ClearML. Пример:
    python train.py --arch resnet50 --seed 0 --tags teacher
    python train.py --arch cnn --cnn_width 32 --n_runs 10 --init_seed 0 --tags student-ce

--n_runs запусков серии отличаются только начальными весами (сиды из --init_seed);
порядок батчей, аугментации и прочую случайность задаёт общий --seed
(см. kd/experiment.py)
"""
import json
import os

from cli.parser import get_parser
from cli.validators import validate_args
from kd.data import DeviceLoader, load_cifar100, stratified_split
from kd.experiment import CrossEntropyExperiment, Loaders, aggregate_runs
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


def report_series(task, args, results: list[dict], run_dir: str) -> None:
    """
    Итог серии в ClearML: single values (среднее и std по запускам),
    артефакты summary и runs (таблица по запускам), веса первого запуска
    """
    import pandas as pd
    from clearml import OutputModel

    agg = aggregate_runs(results)
    report_single_values("val", agg["val"])
    report_single_values("test", agg["test"])
    report_single_values("model", {"n_params": agg["n_params"], "best_val_top1": agg["best_val_top1"]})

    runs = pd.DataFrame([
        {
            "run": r["run"], "model_seed": r["model_seed"], "best_val_top1": r["best_val_top1"],
            **{f"val/{k}": v for k, v in r["val"].items()},
            **{f"test/{k}": v for k, v in r["test"].items()},
        }
        for r in results
    ])
    task.upload_artifact("runs", artifact_object=runs)
    task.upload_artifact("summary", artifact_object={**agg, "args": vars(args)})
    # Только веса первого запуска — хранилище files server ограничено,
    # остальные лежат в <output_dir>/<task_id>/run_<i>/
    OutputModel(task=task, name=f"{args.task_name}-run0", framework="PyTorch").update_weights(
        weights_filename=os.path.join(run_dir, "run_0", "model_final.pt"), auto_delete_file=False
    )


def main():
    args = get_parser().parse_args()
    validate_args(args)
    if args.task_name is None:
        args.task_name = default_task_name(args)

    task = init_task(args) if args.clearml else None

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
    )

    # --- Серия запусков ---
    model_cls, model_kwargs = model_spec(args)
    run_dir = os.path.join(args.output_dir, task.id if task is not None else args.task_name)
    os.makedirs(run_dir, exist_ok=True)
    experiment = CrossEntropyExperiment(
        args, model_cls, model_kwargs, loaders, device, run_dir, task=task
    )
    print(f"{args.task_name}: запусков {args.n_runs}, init_seed {args.init_seed}")
    results = experiment.run(args.n_runs, args.init_seed)

    agg = aggregate_runs(results)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({**agg, "runs": results, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    test = agg["test"]
    std = f" ± {test['top1_std']:.4f}" if "top1_std" in test else ""
    print(f"{agg['n_params'] / 1e6:.2f}M параметров; test top1 {test['top1']:.4f}{std}")

    if task is not None:
        report_series(task, args, results, run_dir)
        task.close()


if __name__ == "__main__":
    main()
