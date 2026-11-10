"""
Общее для сеток через ClearML HyperParameterOptimizer (scripts/kd_grid.py):
черновик задачи train.py для clearml-agent, проверка очереди, зависимости.
scripts/capacity_search.py (работает на master) пока держит свои копии этих функций
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# torch/torchvision в Colab уже стоят (CUDA-сборка) — агент их не переустанавливает
SKIP_PACKAGES = ("torch", "torchvision")
DEFAULT_REPO = "https://github.com/esgon191/cifar100-distillation.git"


def read_packages() -> list[str]:
    """Зависимости из requirements.txt без torch/torchvision"""
    packages = []
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line and line.split("==")[0] not in SKIP_PACKAGES:
                packages.append(line)
    return packages


def check_queue(name: str) -> None:
    """Очередь должна существовать (её создаёт агент: clearml-agent daemon --create-queue)"""
    from clearml.backend_api.session.client import APIClient

    if not APIClient().queues.get_all(name=f"^{name}$"):
        raise SystemExit(
            f'Очередь "{name}" не найдена на сервере ClearML. Сначала запусти агента '
            f"(colab/agent.ipynb: clearml-agent daemon --queue {name} --create-queue)"
        )


def parse_extra_args(items: list[str]) -> list[tuple[str, str]]:
    """--extra_args key=value ... -> [(key, value)]"""
    pairs = []
    for item in items:
        key, value = item.split("=", 1)
        pairs.append((key, value))
    return pairs


def create_train_task(
        project_name: str,
        task_name: str,
        repo: str,
        branch: str,
        argparse_args: list[tuple[str, str]],
        tags: list[str]
):
    """
    Черновик задачи train.py. Параметры — в секции Args, той же, что у
    task.connect(args, name="Args"), поэтому переопределения оптимизатора
    применяются при исполнении на агенте. Списки (nargs="*") передавать строкой
    в формате str(list) — так ClearML сам хранит аргументы-списки
    """
    from clearml import Task

    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        task_type=Task.TaskTypes.training,
        repo=repo,
        branch=branch,
        script="train.py",
        packages=read_packages(),
        argparse_args=argparse_args,
        # train.py сам вызывает Task.init
        add_task_init_call=False,
    )
    task.add_tags(["template", *tags])
    print(f"Базовая задача: {task.id}")
    return task
