"""
Подбор α и T дистилляции по сетке: ClearML HyperParameterOptimizer + GridSearch.
Контроллер — родительская задача, точки сетки (α, T) — её дочерние задачи-клоны
базовой задачи train.py; их можно сравнить в UI ClearML (выделить → Compare).
Каждая точка — серия из --n_runs запусков с разными начальными весами
(--parallel_runs из них одной группой). Цель — val/top1 финальных моделей
(среднее по запускам точки); test в выборе не участвует.

Логиты учителя не зависят от α и T: кэш (kd/teacher_cache.py) строит первая
точка, остальные его используют. --seed, батч и --split_seed — как у бейзлайна
на CE, иначе кэш пересчитается, а сравнение с CE будет некорректным.

Задачи исполняет clearml-agent из очереди (colab/agent.ipynb), контроллер лёгкий —
запускается локально или ячейкой Colab:
    python scripts/kd_grid.py --queue colab-t4

Код агент берёт из репозитория (--repo/--branch): изменения должны быть запушены.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml import Task  # noqa: E402
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer  # noqa: E402
from clearml.backend_api.session import Session  # noqa: E402

from utils.clearml_grid import (  # noqa: E402
    DEFAULT_REPO,
    ROOT,
    check_queue,
    create_train_task,
    parse_extra_args,
)

# Обученный ResNet-50 (шаг 2); то же значение, что cli/parser.py::TEACHER_TASK_ID
TEACHER_TASK_ID = "60b072340b8f4e8aa2ef643d495307bf"


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GridSearch α и T дистилляции в ClearML")
    p.add_argument("--project_name", type=str, default="cifar100-distillation")
    p.add_argument("--queue", type=str, default="colab-t4", help="Очередь clearml-agent")
    p.add_argument("--repo", type=str, default=DEFAULT_REPO)
    p.add_argument("--branch", type=str, default="parallel-runs")
    p.add_argument("--base_task_id", type=str, default=None,
                   help="Готовая базовая задача (иначе создаётся черновик через Task.create)")
    p.add_argument("--task_name", type=str, default="kd-grid-w32-c2", help="Имя задачи-контроллера")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                   help="α в α·CE + (1−α)·T²·KL")
    p.add_argument("--temperatures", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--teacher_task_id", type=str, default=TEACHER_TASK_ID)
    p.add_argument("--cnn_width", type=int, default=32)
    p.add_argument("--cnn_convs_per_stage", type=int, default=2)
    p.add_argument("--n_runs", type=int, default=2, help="Запусков (сидов) на точку сетки")
    p.add_argument("--parallel_runs", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--output_dir", type=str,
                   default="/content/drive/MyDrive/cifar100-distillation/out",
                   help="--output_dir задач (на воркере — Google Drive)")
    p.add_argument("--tags", type=str, nargs="*", default=["student-kd", "kd-grid"])
    p.add_argument("--extra_args", type=str, nargs="*", default=[],
                   help="Доп. аргументы train.py для базовой задачи: --extra_args max_train_samples=512")
    p.add_argument("--create_only", action="store_true",
                   help="Только создать базовую задачу и выйти")
    return p


def base_task_args(args) -> list[tuple[str, str]]:
    """Аргументы train.py базовой задачи: рецепт бейзлайна на CE + KD"""
    pairs = [
        ("arch", "cnn"),
        ("cnn_width", str(args.cnn_width)),
        ("cnn_convs_per_stage", str(args.cnn_convs_per_stage)),
        # Явной строкой: у --teacher_task_id nargs="?", без значения клон получил бы ""
        ("teacher_task_id", args.teacher_task_id),
        # Перебираемые параметры — явно, чтобы они были в секции Args базовой задачи
        ("kd_alpha", str(args.alphas[0])),
        ("kd_temperature", str(args.temperatures[0])),
        ("n_runs", str(args.n_runs)),
        ("parallel_runs", str(args.parallel_runs)),
        ("init_seed", "0"),
        ("seed", "0"),
        ("num_train_epochs", str(args.num_train_epochs)),
        ("learning_rate", "0.1"),
        ("weight_decay", "5e-4"),
        ("warmup_epochs", "1"),
        ("per_device_train_batch_size", "128"),
        ("save_every", "5"),
        ("dataset_name", "cifar100"),
        ("output_dir", args.output_dir),
        # Список (nargs="*") — в формате, в котором ClearML сам хранит аргументы-списки
        ("tags", str(list(args.tags))),
    ]
    return pairs + parse_extra_args(args.extra_args)


def collect_results(task_ids: list[str]) -> "pd.DataFrame":
    """Таблица точек сетки: α, T и single values финальных моделей (только val и train)"""
    import pandas as pd

    keys = ("top1", "top1_std", "top5", "ce", "kl", "loss")
    rows = []
    for t in Task.get_tasks(task_ids=task_ids):
        params = t.get_parameters_as_dict().get("Args", {})
        values = t.get_reported_single_values()
        row = {
            "task_id": t.id,
            "status": t.get_status(),
            "kd_alpha": float(params.get("kd_alpha", "nan")),
            "kd_temperature": float(params.get("kd_temperature", "nan")),
        }
        row.update({f"val/{k}": values.get(f"val/{k}") for k in keys})
        row.update({f"train/{k}": values.get(f"train/{k}") for k in ("top1", "ce", "kl")})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("val/top1", ascending=False)


def main():
    args = get_parser().parse_args()
    if not args.create_only:
        check_queue(args.queue)

    # Контроллер — родительская задача типа optimizer; артефакты на files server (не S3)
    controller = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.optimizer,
        tags=list(args.tags),
        output_uri=Session.get_files_server_host(),
        reuse_last_task_id=False,
    )

    base_task_id = args.base_task_id or create_train_task(
        args.project_name, f"{args.task_name}-base", args.repo, args.branch,
        base_task_args(args), list(args.tags)
    ).id
    if args.create_only:
        controller.close()
        return

    n_points = len(args.alphas) * len(args.temperatures)
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            DiscreteParameterRange("Args/kd_alpha", values=args.alphas),
            DiscreteParameterRange("Args/kd_temperature", values=args.temperatures),
        ],
        # Последнее значение кривой val/top1 — среднее по запускам точки на финальной
        # модели (train.py::report_series отправляет кривые в конце серии)
        objective_metric_title="top1",
        objective_metric_series="val",
        objective_metric_sign="max",
        optimizer_class=GridSearch,
        execution_queue=args.queue,
        # Один T4: точки идут по очереди
        max_number_of_concurrent_tasks=1,
        total_max_jobs=n_points,
        spawn_project=None,
        pool_period_min=1.0,
    )
    # start(): контроллер работает в этом процессе, а точки сетки ставятся в очередь
    # агенту. start_locally() не подходит — он исполнил бы их здесь же, на этой машине
    optimizer.start()
    optimizer.wait()
    optimizer.stop()

    task_ids = [t.id for t in optimizer.get_top_experiments(top_k=n_points)]
    table = collect_results(task_ids)
    print(table.to_string(index=False))
    out_dir = args.output_dir if os.path.isdir(args.output_dir) else os.path.join(ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.task_name}.csv")
    table.to_csv(path, index=False)
    print(f"Таблица сетки: {path}")
    controller.upload_artifact("results", artifact_object=table)
    controller.close()


if __name__ == "__main__":
    main()
