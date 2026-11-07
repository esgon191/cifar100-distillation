"""
Подбор ёмкости ученика по сетке: ClearML HyperParameterOptimizer + GridSearch.
Каждая конфигурация (cnn_width × cnn_convs_per_stage) обучается один раз
(--n_runs 1) задачей-клоном базовой задачи; задачи исполняет clearml-agent
из очереди (в Colab — colab/agent.ipynb). Цель — val/top1 финальной модели.

Контроллер лёгкий (только ставит задачи в очередь и опрашивает их),
запускается локально или ячейкой Colab:
    python scripts/capacity_search.py --queue colab-t4 \
        --output_dir /content/drive/MyDrive/cifar100-distillation/out

Код для задач агент берёт из репозитория (--repo/--branch): перед запуском
изменения должны быть запушены.
"""
import argparse
import os

from clearml import Task
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer
from clearml.backend_api.session import Session

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# torch/torchvision в Colab уже стоят (CUDA-сборка) — агент их не переустанавливает
SKIP_PACKAGES = ("torch", "torchvision")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GridSearch ёмкости CNN-ученика в ClearML")
    p.add_argument("--project_name", type=str, default="cifar100-distillation")
    p.add_argument("--queue", type=str, default="colab-t4", help="Очередь clearml-agent")
    p.add_argument("--repo", type=str, default="https://github.com/esgon191/cifar100-distillation.git")
    p.add_argument("--branch", type=str, default="master")
    p.add_argument("--base_task_id", type=str, default=None,
                   help="Готовая базовая задача (иначе создаётся черновик через Task.create)")
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 48, 64])
    p.add_argument("--convs", type=int, nargs="+", default=[1, 2])
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--output_dir", type=str,
                   default="/content/drive/MyDrive/cifar100-distillation/out",
                   help="--output_dir задач (на воркере — Google Drive)")
    p.add_argument("--tags", type=str, nargs="*", default=["student-ce", "capacity"])
    p.add_argument("--extra_args", type=str, nargs="*", default=[],
                   help="Доп. аргументы train.py для базовой задачи: --extra_args max_train_samples=512")
    p.add_argument("--create_only", action="store_true",
                   help="Только создать базовую задачу и выйти")
    return p


def read_packages() -> list[str]:
    """Зависимости из requirements.txt без torch/torchvision"""
    packages = []
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line and line.split("==")[0] not in SKIP_PACKAGES:
                packages.append(line)
    return packages


def create_base_task(args) -> Task:
    """
    Черновик задачи train.py для ученика. Параметры — в секции Args, той же,
    что у task.connect(args, name="Args"), поэтому переопределения оптимизатора
    применяются при исполнении на агенте
    """
    argparse_args = [
        ("arch", "cnn"),
        # Перебираемые параметры — явно, чтобы они были в секции Args базовой задачи
        ("cnn_width", str(args.widths[0])),
        ("cnn_convs_per_stage", str(args.convs[0])),
        ("dataset_name", "cifar100"),
        ("num_train_epochs", str(args.num_train_epochs)),
        ("n_runs", "1"),
        ("output_dir", args.output_dir),
        # Список (nargs="*") — в формате, в котором ClearML сам хранит аргументы-списки
        ("tags", str(list(args.tags))),
    ]
    for item in args.extra_args:
        key, value = item.split("=", 1)
        argparse_args.append((key, value))

    task = Task.create(
        project_name=args.project_name,
        task_name="student-capacity-base",
        task_type=Task.TaskTypes.training,
        repo=args.repo,
        branch=args.branch,
        script="train.py",
        packages=read_packages(),
        argparse_args=argparse_args,
        # train.py сам вызывает Task.init
        add_task_init_call=False,
    )
    task.add_tags(["template", *args.tags])
    print(f"Базовая задача: {task.id}")
    return task


def collect_results(task_ids: list[str]) -> "pd.DataFrame":
    """Таблица конфигураций сетки: параметры, single values и время эпохи"""
    import pandas as pd

    rows = []
    for t in Task.get_tasks(task_ids=task_ids):
        params = t.get_parameters_as_dict().get("Args", {})
        values = t.get_reported_single_values()
        scalars = t.get_last_scalar_metrics()
        rows.append({
            "task_id": t.id,
            "status": t.get_status(),
            "cnn_width": int(params.get("cnn_width", 0)),
            "cnn_convs_per_stage": int(params.get("cnn_convs_per_stage", 0)),
            "n_params": values.get("model/n_params"),
            "val/top1": values.get("val/top1"),
            "val/ce": values.get("val/ce"),
            "val/precision": values.get("val/precision"),
            "best_val_top1": values.get("model/best_val_top1"),
            "train/top1": scalars.get("top1", {}).get("train", {}).get("last"),
            "epoch_time": scalars.get("epoch_time", {}).get("train", {}).get("last"),
        })
    return pd.DataFrame(rows).sort_values("n_params")


def check_queue(name: str) -> None:
    """Очередь должна существовать (её создаёт агент: clearml-agent daemon --create-queue)"""
    from clearml.backend_api.session.client import APIClient

    if not APIClient().queues.get_all(name=f"^{name}$"):
        raise SystemExit(
            f'Очередь "{name}" не найдена на сервере ClearML. Сначала запусти агента '
            f"(colab/agent.ipynb: clearml-agent daemon --queue {name} --create-queue)"
        )


def main():
    args = get_parser().parse_args()
    if not args.create_only:
        check_queue(args.queue)

    # Контроллер — отдельная задача типа optimizer; артефакты на files server (не S3)
    controller = Task.init(
        project_name=args.project_name,
        task_name="student-capacity-search",
        task_type=Task.TaskTypes.optimizer,
        tags=list(args.tags),
        output_uri=Session.get_files_server_host(),
        reuse_last_task_id=False,
    )

    base_task_id = args.base_task_id or create_base_task(args).id
    if args.create_only:
        controller.close()
        return

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            DiscreteParameterRange("Args/cnn_width", values=args.widths),
            DiscreteParameterRange("Args/cnn_convs_per_stage", values=args.convs),
        ],
        # Последнее значение val/top1 — финальная модель (конец косинуса)
        objective_metric_title="top1",
        objective_metric_series="val",
        objective_metric_sign="max",
        optimizer_class=GridSearch,
        execution_queue=args.queue,
        max_number_of_concurrent_tasks=1,
        total_max_jobs=len(args.widths) * len(args.convs),
        spawn_project=None,
        pool_period_min=1.0,
    )
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()

    task_ids = [t.id for t in optimizer.get_top_experiments(top_k=len(args.widths) * len(args.convs))]
    table = collect_results(task_ids)
    print(table.to_string(index=False))
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    table.to_csv(os.path.join(ROOT, "out", "capacity_search.csv"), index=False)
    controller.upload_artifact("results", artifact_object=table)
    controller.close()


if __name__ == "__main__":
    main()
