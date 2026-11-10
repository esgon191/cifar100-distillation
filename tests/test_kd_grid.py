"""Сетка α и T (scripts/kd_grid.py): аргументы базовой задачи — валидные аргументы train.py"""
import ast
import importlib.util
import os

from cli.parser import TEACHER_TASK_ID, get_parser
from cli.validators import validate_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kd_grid():
    spec = importlib.util.spec_from_file_location("kd_grid", os.path.join(ROOT, "scripts", "kd_grid.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_task_args_parse_as_train_args():
    kd_grid = _kd_grid()
    assert kd_grid.TEACHER_TASK_ID == TEACHER_TASK_ID

    grid_args = kd_grid.get_parser().parse_args(["--alphas", "0.3", "0.7", "--temperatures", "4"])
    argv = []
    for key, value in kd_grid.base_task_args(grid_args):
        if key == "tags":
            argv += ["--tags", *ast.literal_eval(value)]
        else:
            argv += [f"--{key}", value]
    args = get_parser().parse_args(argv)
    validate_args(args)

    assert args.teacher_task_id == TEACHER_TASK_ID
    assert (args.kd_alpha, args.kd_temperature) == (0.3, 4.0)
    assert (args.n_runs, args.parallel_runs, args.seed, args.init_seed) == (2, 2, 0, 0)
    assert args.tags == ["student-kd", "kd-grid"]
