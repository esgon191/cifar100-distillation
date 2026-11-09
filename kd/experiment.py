"""
Эксперимент: многократное обучение одной модели из разных начальных состояний.

Случайность разделена на два источника:
- сид инициализаций (--init_seed) детерминирует последовательность начальных
  состояний — по одному сиду весов на запуск (numpy SeedSequence.spawn);
  все сиды известны заранее;
- глобальный сид (--seed) — всё остальное: порядок батчей, аугментации, прочие
  случайные операции. Он одинаков во всех запусках серии.

Поэтому разброс метрик между запусками серии отражает только начальное состояние,
а батчи у всех запусков одни и те же: --parallel_runs запусков обучаются
синхронно одной группой (kd/stacked.py), результат запуска от группы не зависит.

Каждый запуск сохраняет на диск (<run_dir>/run_<i>/) историю обучения, финальный
замер всех метрик на train/val/test, логиты финальной модели на val/test и её веса.
Сводные таблицы серии собираются из этих файлов (collect_series).
Конкретный эксперимент задаёт целевую функцию (и, для KD, учителя) в подклассе
"""
import csv
import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from kd.data import DeviceLoader
from kd.models import count_params
from kd.stacked import stack_models
from kd.trainer import (
    CrossEntropyObjective,
    build_optimizer,
    build_scheduler,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)

# Части данных финального замера и части, для которых сохраняются логиты
FINAL_SPLITS = ("train", "val", "test")
LOGIT_SPLITS = ("val", "test")
# История обучения — long-формат: строка = (запуск, эпоха, часть, метрика, значение)
HISTORY_FIELDS = ("run", "epoch", "split", "metric", "value")
# Файлы завершённого запуска и сводные файлы серии
RUN_FILES = ("final.json", "history.csv", "logits.npz", "model_final.pt")
SERIES_FILES = ("series.json", "final_metrics.csv", "test_metrics.csv", "history.csv")


@dataclass
class Loaders:
    """
    Загрузчики частей данных (общие для всех запусков серии).
    train_eval — те же примеры train без аугментаций, для финального замера
    """
    train: DeviceLoader
    val: DeviceLoader
    test: DeviceLoader
    train_eval: DeviceLoader | None = None


def init_seeds(n_runs: int, init_seed: int) -> list[int]:
    """
    Сиды инициализации весов для n_runs запусков. Сид запуска i зависит
    только от (init_seed, i): при увеличении n_runs первые запуски не меняются
    """
    children = np.random.SeedSequence(init_seed).spawn(n_runs)
    return [int(c.generate_state(1)[0]) for c in children]


def _git_commit() -> str | None:
    """Коммит кода, которым обучалась серия (если запуск из git-репозитория)"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: str, obj) -> None:
    """Атомарная запись json: временный файл и переименование"""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


class Experiment(ABC):
    """
    Обучает модель model_cls(**model_kwargs) заданное число раз из разных
    начальных состояний (run), группами по --parallel_runs запусков.
    Завершённый запуск — тот, у которого есть run_<i>/final.json; при повторном
    вызове такие запуски пропускаются, прерванная группа продолжается с чекпоинта
    """

    def __init__(
            self,
            args: Namespace,
            model_cls: Callable[..., nn.Module],
            model_kwargs: dict,
            loaders: Loaders,
            device: torch.device,
            run_dir: str,
            task=None
    ):
        self.args = args
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.loaders = loaders
        self.device = device
        self.run_dir = run_dir
        self.task = task
        self.amp = args.fp16 and device.type == "cuda"
        self.channels_last = device.type == "cuda"
        self.parallel_runs = getattr(args, "parallel_runs", 1)
        self.n_runs = 1

    # --- Точки расширения ---

    @abstractmethod
    def build_objective(self):
        """Целевая функция группы: (logits [K,B,C], targets, teacher_logits) -> (loss [K], компоненты)"""

    def teacher(self) -> nn.Module | None:
        """Учитель для KD (прогоняется онлайн, один раз на батч для всей группы); по умолчанию нет"""
        return None

    # --- Модель ---

    def build_model(self, model_seed: int) -> nn.Module:
        """
        Модель с начальным состоянием, заданным model_seed. Сразу после
        создания глобальные генераторы возвращаются к глобальному сиду,
        так что от model_seed зависит только инициализация весов
        """
        torch.manual_seed(model_seed)
        model = self.model_cls(**self.model_kwargs)
        set_seed(self.args.seed, deterministic=self.args.deterministic)
        model = model.to(self.device)
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    def count_model_params(self) -> int:
        """Число параметров одной модели (без расхода глобального генератора)"""
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            return count_params(self.model_cls(**self.model_kwargs))

    # --- Файлы ---

    def _run_path(self, run_idx: int, name: str) -> str:
        return os.path.join(self.run_dir, f"run_{run_idx}", name)

    def _group_dir(self, run_ids: list[int]) -> str:
        return os.path.join(self.run_dir, f"group_{run_ids[0]}")

    def is_done(self, run_idx: int) -> bool:
        return os.path.isfile(self._run_path(run_idx, "final.json"))

    def load_final(self, run_idx: int) -> dict:
        with open(self._run_path(run_idx, "final.json")) as f:
            return json.load(f)

    def _find_checkpoint(self, run_ids: list[int]) -> str | None:
        """Чекпоинт группы на диске, иначе артефакт ClearML checkpoint_group<первый запуск>"""
        path = os.path.join(self._group_dir(run_ids), "checkpoint.pt")
        if os.path.isfile(path):
            return path
        name = f"checkpoint_group{run_ids[0]}"
        if self.task is not None and name in self.task.artifacts:
            return self.task.artifacts[name].get_local_copy()
        return None

    def _reset_history(self, run_idx: int, keep_until_epoch: int) -> None:
        """
        Оставляет в истории запуска строки эпох <= keep_until_epoch: при продолжении
        с чекпоинта эпохи после него будут пройдены и записаны заново
        """
        path = self._run_path(run_idx, "history.csv")
        if not os.path.isfile(path):
            return
        if keep_until_epoch <= 0:
            os.remove(path)
            return
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= keep_until_epoch]
        with open(path + ".tmp", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(path + ".tmp", path)

    def _append_history(self, run_idx: int, epoch: int, split: str, metrics: dict) -> None:
        path = self._run_path(run_idx, "history.csv")
        new = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(HISTORY_FIELDS)
            for metric, value in metrics.items():
                writer.writerow((run_idx, epoch, split, metric, value))

    def write_series_meta(self, model_seeds: list[int], init_seed: int) -> None:
        """series.json: параметры серии, сиды всех запусков, размер модели, коммит"""
        _write_json(os.path.join(self.run_dir, "series.json"), {
            "task_name": self.args.task_name,
            "task_id": self.task.id if self.task is not None else None,
            "n_runs": len(model_seeds),
            "init_seed": init_seed,
            "seed": self.args.seed,
            "parallel_runs": self.parallel_runs,
            "model_seeds": model_seeds,
            "n_params": self.count_model_params(),
            "git_commit": _git_commit(),
            "args": vars(self.args),
        })

    # --- Обучение ---

    def _plan_groups(self, pending: list[int]) -> list[list[int]]:
        """
        Разбиение незавершённых запусков на группы. Группа с сохранённым
        составом (group_<i>/group.json) восстанавливается как была — её чекпоинт
        содержит именно эти модели; остальные запуски идут подряд по parallel_runs
        """
        groups, used = [], set()
        if self.args.resume and os.path.isdir(self.run_dir):
            for name in sorted(os.listdir(self.run_dir)):
                meta = os.path.join(self.run_dir, name, "group.json")
                if not (name.startswith("group_") and os.path.isfile(meta)):
                    continue
                with open(meta) as f:
                    ids = json.load(f)["run_ids"]
                if set(ids) <= set(pending) - used:
                    groups.append(ids)
                    used.update(ids)
        rest = [i for i in pending if i not in used]
        k = self.parallel_runs
        groups += [rest[i:i + k] for i in range(0, len(rest), k)]
        return sorted(groups, key=lambda g: g[0])

    def _log_line(self, run_ids: list[int], epoch: int, train: list[dict], val: list[dict] | None,
                  epoch_time: float) -> str:
        """Строка прогресса эпохи в консоли (её сохраняет и ClearML)"""
        total = self.args.num_train_epochs
        if len(run_ids) == 1:
            prefix = f"[run {run_ids[0]}] " if self.n_runs > 1 else ""
            line = (f"{prefix}[{epoch}/{total}] train loss {train[0]['loss']:.4f} "
                    f"top1 {train[0]['top1']:.4f} ({epoch_time:.1f}s)")
            if val:
                line += f" | val loss {val[0]['loss']:.4f} top1 {val[0]['top1']:.4f}"
            return line
        med = lambda ms, key: float(np.median([m[key] for m in ms]))
        line = (f"[runs {run_ids[0]}..{run_ids[-1]}] [{epoch}/{total}] "
                f"train loss med {med(train, 'loss'):.4f} top1 med {med(train, 'top1'):.4f} "
                f"({epoch_time:.1f}s)")
        if val:
            top1 = [m["top1"] for m in val]
            line += (f" | val top1 med {np.median(top1):.4f} "
                     f"[{min(top1):.4f}..{max(top1):.4f}]")
        return line

    def train_group(self, run_ids: list[int], model_seeds: list[int]) -> list[dict]:
        """
        Группа запусков: K моделей (каждая — со своим начальным состоянием)
        обучаются синхронно на общих батчах, затем финальные модели замеряются
        на train/val/test и сохраняются. Возвращает final каждого запуска
        """
        args = self.args
        loaders = self.loaders
        models = [self.build_model(seed) for seed in model_seeds]
        n_params = count_params(models[0])
        model = stack_models(models, channels_last=self.channels_last)
        del models
        teacher = self.teacher()
        objective = self.build_objective()

        steps_per_epoch = len(loaders.train)
        optimizer = build_optimizer(
            model, args.learning_rate, args.weight_decay, args.momentum, args.nesterov,
            decay_norm_bias=args.weight_decay_norm_bias
        )
        scheduler = build_scheduler(
            optimizer,
            total_steps=args.num_train_epochs * steps_per_epoch,
            warmup_steps=args.warmup_epochs * steps_per_epoch
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        group_dir = self._group_dir(run_ids)
        ckpt_path = os.path.join(group_dir, "checkpoint.pt")
        os.makedirs(group_dir, exist_ok=True)
        _write_json(os.path.join(group_dir, "group.json"),
                    {"run_ids": run_ids, "model_seeds": model_seeds})
        for run_idx in run_ids:
            os.makedirs(os.path.dirname(self._run_path(run_idx, "")), exist_ok=True)

        start_epoch, best_val_top1 = 0, [0.0] * len(run_ids)
        if args.resume:
            found = self._find_checkpoint(run_ids)
            if found is not None:
                start_epoch, extra = load_checkpoint(
                    found, model, optimizer, scheduler, scaler, self.device
                )
                if extra.get("run_ids") != run_ids:
                    raise RuntimeError(
                        f"Чекпоинт {found} — для запусков {extra.get('run_ids')}, а не {run_ids}"
                    )
                best_val_top1 = extra["best_val_top1"]
                print(f"[runs {run_ids}] Продолжение с эпохи {start_epoch} ({found})")
        for run_idx in run_ids:
            self._reset_history(run_idx, start_epoch)

        for epoch in range(start_epoch, args.num_train_epochs):
            last_epoch = epoch == args.num_train_epochs - 1
            loaders.train.set_epoch(epoch)
            t0 = time.time()
            train_metrics, skipped = train_one_epoch(
                model, loaders.train, optimizer, scheduler, scaler, objective,
                self.device, self.amp, teacher=teacher
            )
            epoch_time = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            for k, run_idx in enumerate(run_ids):
                self._append_history(run_idx, epoch + 1, "train", {
                    **train_metrics[k], "lr": lr, "epoch_time": epoch_time,
                    "skipped_steps": skipped,
                })

            val_metrics = None
            if (epoch + 1) % args.eval_every == 0 or last_epoch:
                val_metrics = evaluate(
                    model, loaders.val, objective, self.device, self.amp, teacher=teacher
                )
                for k, run_idx in enumerate(run_ids):
                    best_val_top1[k] = max(best_val_top1[k], val_metrics[k]["top1"])
                    self._append_history(run_idx, epoch + 1, "val", val_metrics[k])
            print(self._log_line(run_ids, epoch + 1, train_metrics, val_metrics, epoch_time))

            if (epoch + 1) % args.save_every == 0 or last_epoch:
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler, epoch,
                    extra={"run_ids": run_ids, "model_seeds": model_seeds,
                           "best_val_top1": best_val_top1}
                )
                if self.task is not None and args.upload_checkpoint:
                    self.task.upload_artifact(
                        f"checkpoint_group{run_ids[0]}", artifact_object=ckpt_path,
                        wait_on_upload=True
                    )

        finals = self._final_measure(model, run_ids, model_seeds, n_params, best_val_top1,
                                     objective, teacher)
        # Группа завершена: её чекпоинт больше не нужен (веса каждой модели — в model_final.pt)
        shutil.rmtree(group_dir, ignore_errors=True)
        return finals

    def _final_measure(self, model: nn.Module, run_ids: list[int], model_seeds: list[int],
                       n_params: int, best_val_top1: list[float], objective, teacher) -> list[dict]:
        """
        Финальный замер: финальные модели (конец косинуса) в режиме eval без
        аугментаций на train, val и test — все метрики и компоненты лосса.
        Сохраняет логиты val/test, веса и final.json каждого запуска
        """
        loaders = self.loaders
        split_loaders = {"train": loaders.train_eval, "val": loaders.val, "test": loaders.test}
        measured, logits = {}, {}
        for split in FINAL_SPLITS:
            loader = split_loaders[split]
            if loader is None:
                continue
            if split in LOGIT_SPLITS:
                measured[split], logits[split] = evaluate(
                    model, loader, objective, self.device, self.amp, teacher=teacher,
                    return_logits=True
                )
            else:
                measured[split] = evaluate(model, loader, objective, self.device, self.amp,
                                           teacher=teacher)

        finals = []
        for k, (run_idx, seed) in enumerate(zip(run_ids, model_seeds)):
            final = {
                "run": run_idx,
                "model_seed": seed,
                "n_params": n_params,
                "best_val_top1": best_val_top1[k],
                **{split: metrics[k] for split, metrics in measured.items()},
            }
            np.savez(
                self._run_path(run_idx, "logits.npz"),
                **{split: logits[split][k].numpy() for split in logits},
                **{f"y_{split}": split_loaders[split].y.cpu().numpy() for split in logits},
            )
            torch.save(
                {"model": model.unstack(k), "model_seed": seed, "args": vars(self.args)},
                self._run_path(run_idx, "model_final.pt")
            )
            # final.json пишется последним: по нему запуск считается завершённым
            _write_json(self._run_path(run_idx, "final.json"), final)
            finals.append(final)
            if "test" in final:
                print(f"[run {run_idx}] финальная модель: test top1 {final['test']['top1']:.4f} "
                      f"top5 {final['test']['top5']:.4f}")
        return finals

    def run(self, n_runs: int, init_seed: int) -> list[dict]:
        """
        Серия из n_runs запусков с начальными состояниями из init_seeds(n_runs, init_seed).
        Возвращает final всех запусков (по порядку)
        """
        self.n_runs = n_runs
        seeds = init_seeds(n_runs, init_seed)
        os.makedirs(self.run_dir, exist_ok=True)
        self.write_series_meta(seeds, init_seed)

        done = [i for i in range(n_runs) if self.args.resume and self.is_done(i)]
        for run_idx in done:
            print(f"[run {run_idx}] уже завершён — пропуск")
        pending = [i for i in range(n_runs) if i not in done]
        for group in self._plan_groups(pending):
            self.train_group(group, [seeds[i] for i in group])
        return [self.load_final(i) for i in range(n_runs)]


class CrossEntropyExperiment(Experiment):
    """Обучение только на метках (CE): учитель и бейзлайн ученика"""

    def build_objective(self):
        return CrossEntropyObjective(self.args.label_smoothing)


# --- Сборка серии из файлов ---

def missing_series_files(run_dir: str, n_runs: int, with_tables: bool = True) -> list[str]:
    """Файлы серии, которых нет на диске (пустой список — всё на месте)"""
    expected = [os.path.join(run_dir, f"run_{i}", name) for i in range(n_runs) for name in RUN_FILES]
    names = SERIES_FILES if with_tables else ("series.json",)
    expected += [os.path.join(run_dir, name) for name in names]
    return [path for path in expected if not os.path.isfile(path)]


def verify_series_files(run_dir: str, n_runs: int, with_tables: bool = True) -> None:
    """Проверяет, что все файлы серии сохранены на диск; иначе — ошибка со списком"""
    missing = missing_series_files(run_dir, n_runs, with_tables)
    if missing:
        listed = "\n  ".join(missing)
        raise FileNotFoundError(f"В {run_dir} не хватает файлов серии:\n  {listed}")


def collect_series(run_dir: str, n_runs: int) -> dict[str, pd.DataFrame]:
    """
    Сводные таблицы серии из файлов запусков; сохраняются в run_dir:
    - final_metrics.csv — финальный замер, long: run, model_seed, split, metric, value;
    - test_metrics.csv — финальный замер на test, строка = запуск;
    - history.csv — история обучения всех запусков, long: run, epoch, split, metric, value
    """
    verify_series_files(run_dir, n_runs, with_tables=False)
    final_rows, test_rows, histories = [], [], []
    for i in range(n_runs):
        with open(os.path.join(run_dir, f"run_{i}", "final.json")) as f:
            final = json.load(f)
        for split in FINAL_SPLITS:
            for metric, value in final.get(split, {}).items():
                final_rows.append({"run": i, "model_seed": final["model_seed"],
                                   "split": split, "metric": metric, "value": value})
        test_rows.append({"run": i, "model_seed": final["model_seed"], **final.get("test", {})})
        histories.append(pd.read_csv(os.path.join(run_dir, f"run_{i}", "history.csv")))

    tables = {
        "final_metrics": pd.DataFrame(final_rows),
        "test_metrics": pd.DataFrame(test_rows),
        "history": pd.concat(histories, ignore_index=True),
    }
    for name, table in tables.items():
        path = os.path.join(run_dir, f"{name}.csv")
        table.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    return tables


def aggregate_final(final_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Среднее и (при n > 1) несмещённое std финальных метрик по запускам:
    {split: {metric: mean, metric_std: std}}
    """
    out = {}
    for split, frame in final_metrics.groupby("split"):
        agg = {}
        for metric, values in frame.groupby("metric")["value"]:
            agg[metric] = float(values.mean())
            if len(values) > 1:
                agg[f"{metric}_std"] = float(values.std(ddof=1))
        out[split] = agg
    return out
