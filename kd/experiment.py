"""
Эксперимент: многократное обучение одной модели из разных начальных состояний.

Случайность разделена на два источника:
- сид инициализаций (--init_seed) детерминирует последовательность начальных
  состояний — по одному сиду весов на запуск (numpy SeedSequence.spawn);
- глобальный сид (--seed) — всё остальное: порядок батчей, аугментации, прочие
  случайные операции. Он одинаков во всех запусках серии.

Поэтому разброс метрик между запусками серии отражает только начальное состояние.
Конкретный эксперимент задаёт целевую функцию (и, для KD, учителя) в подклассе
"""
import json
import os
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from kd.data import DeviceLoader
from kd.models import count_params
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

# Метрики, по которым серия агрегируется в single values (среднее и std)
SUMMARY_METRICS = ("top1", "top5", "precision", "recall", "ce", "loss")


@dataclass
class Loaders:
    """Загрузчики частей данных (общие для всех запусков серии)"""
    train: DeviceLoader
    val: DeviceLoader
    test: DeviceLoader


def init_seeds(n_runs: int, init_seed: int) -> list[int]:
    """
    Сиды инициализации весов для n_runs запусков. Сид запуска i зависит
    только от (init_seed, i): при увеличении n_runs первые запуски не меняются
    """
    children = np.random.SeedSequence(init_seed).spawn(n_runs)
    return [int(c.generate_state(1)[0]) for c in children]


class Experiment(ABC):
    """
    Обучает модель model_cls(**model_kwargs) заданное число раз из разных
    начальных состояний (run). Каждый запуск пишет в <run_dir>/run_<i>/
    checkpoint.pt, summary.json, model_final.pt; завершённые запуски
    при повторном вызове пропускаются, прерванный продолжается с чекпоинта
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
        self.n_runs = 1

    # --- Точки расширения ---

    @abstractmethod
    def build_objective(self):
        """Целевая функция: (logits, targets, teacher_logits) -> (loss, компоненты)"""

    def teacher(self) -> nn.Module | None:
        """Учитель для KD (прогоняется онлайн на тех же батчах); по умолчанию нет"""
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

    # --- Логирование ---

    def _series(self, split: str, run_idx: int) -> str:
        # Одиночный запуск — series train/val, как у учителя (на них смотрит GridSearch)
        return split if self.n_runs == 1 else f"{split}/run{run_idx}"

    def _report(self, split: str, run_idx: int, metrics: dict, epoch: int) -> None:
        if self.task is not None:
            from utils.clearml_utils import report_split_metrics
            report_split_metrics(self._series(split, run_idx), metrics, epoch)

    # --- Обучение ---

    def _run_path(self, run_idx: int, name: str) -> str:
        return os.path.join(self.run_dir, f"run_{run_idx}", name)

    def _find_checkpoint(self, run_idx: int) -> str | None:
        """Локальный чекпоинт запуска, иначе артефакт ClearML checkpoint_run<i>"""
        path = self._run_path(run_idx, "checkpoint.pt")
        if os.path.isfile(path):
            return path
        name = f"checkpoint_run{run_idx}"
        if self.task is not None and name in self.task.artifacts:
            return self.task.artifacts[name].get_local_copy()
        return None

    def _find_summary(self, run_idx: int) -> dict | None:
        """Итог завершённого запуска: локальный summary.json или артефакт summary_run<i>"""
        path = self._run_path(run_idx, "summary.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
        name = f"summary_run{run_idx}"
        if self.task is not None and name in self.task.artifacts:
            return self.task.artifacts[name].get()
        return None

    def train_once(self, run_idx: int, model_seed: int) -> dict:
        """
        Один запуск: обучение с нуля (или продолжение с чекпоинта) и оценка
        финальной модели на val и test. Возвращает summary запуска
        """
        args = self.args
        loaders = self.loaders
        model = self.build_model(model_seed)
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

        ckpt_path = self._run_path(run_idx, "checkpoint.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        start_epoch, best_val_top1 = 0, 0.0
        if args.resume:
            found = self._find_checkpoint(run_idx)
            if found is not None:
                start_epoch, extra = load_checkpoint(
                    found, model, optimizer, scheduler, scaler, self.device
                )
                best_val_top1 = extra.get("best_val_top1", 0.0)
                print(f"[run {run_idx}] Продолжение с эпохи {start_epoch} ({found})")

        prefix = f"[run {run_idx}] " if self.n_runs > 1 else ""
        val_metrics = {}
        for epoch in range(start_epoch, args.num_train_epochs):
            last_epoch = epoch == args.num_train_epochs - 1
            loaders.train.set_epoch(epoch)
            t0 = time.time()
            train_metrics = train_one_epoch(
                model, loaders.train, optimizer, scheduler, scaler, objective,
                self.device, self.amp, teacher=teacher
            )
            train_metrics["lr"] = scheduler.get_last_lr()[0]
            train_metrics["epoch_time"] = time.time() - t0
            self._report("train", run_idx, train_metrics, epoch + 1)

            log_line = (
                f"{prefix}[{epoch + 1}/{args.num_train_epochs}] "
                f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.4f} "
                f"({train_metrics['epoch_time']:.1f}s)"
            )
            if (epoch + 1) % args.eval_every == 0 or last_epoch:
                val_metrics = evaluate(
                    model, loaders.val, objective, self.device, self.amp, teacher=teacher
                )
                best_val_top1 = max(best_val_top1, val_metrics["top1"])
                log_line += f" | val loss {val_metrics['loss']:.4f} top1 {val_metrics['top1']:.4f}"
                self._report("val", run_idx, val_metrics, epoch + 1)
            print(log_line)

            if (epoch + 1) % args.save_every == 0 or last_epoch:
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler, epoch,
                    extra={"best_val_top1": best_val_top1}
                )
                if self.task is not None and args.upload_checkpoint:
                    self.task.upload_artifact(
                        f"checkpoint_run{run_idx}", artifact_object=ckpt_path,
                        wait_on_upload=True
                    )

        # Итог — финальная модель (конец косинусного графика)
        if not val_metrics:
            val_metrics = evaluate(model, loaders.val, objective, self.device, self.amp, teacher=teacher)
        test_metrics = evaluate(model, loaders.test, objective, self.device, self.amp, teacher=teacher)
        summary = {
            "run": run_idx,
            "model_seed": model_seed,
            "n_params": count_params(model),
            "val": val_metrics,
            "best_val_top1": best_val_top1,
            "test": test_metrics,
        }
        print(f"{prefix}test top1 {test_metrics['top1']:.4f} top5 {test_metrics['top5']:.4f}")

        torch.save(
            {"model": model.state_dict(), "model_seed": model_seed, "args": vars(args)},
            self._run_path(run_idx, "model_final.pt")
        )
        # summary.json пишется последним: по нему запуск считается завершённым
        summary_path = self._run_path(run_idx, "summary.json")
        with open(summary_path + ".tmp", "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        os.replace(summary_path + ".tmp", summary_path)
        if self.task is not None:
            self.task.upload_artifact(f"summary_run{run_idx}", artifact_object=summary)
        return summary

    def run(self, n_runs: int, init_seed: int) -> list[dict]:
        """
        Серия из n_runs запусков с начальными состояниями из init_seeds(n_runs, init_seed).
        Возвращает summary всех запусков (по порядку)
        """
        self.n_runs = n_runs
        results = []
        for run_idx, model_seed in enumerate(init_seeds(n_runs, init_seed)):
            summary = self._find_summary(run_idx) if self.args.resume else None
            if summary is not None:
                print(f"[run {run_idx}] уже завершён — пропуск")
            else:
                summary = self.train_once(run_idx, model_seed)
            results.append(summary)
        return results


class CrossEntropyExperiment(Experiment):
    """Обучение только на метках (CE): учитель и бейзлайн ученика"""

    def build_objective(self):
        return CrossEntropyObjective(self.args.label_smoothing)


def aggregate_runs(results: list[dict]) -> dict:
    """
    Агрегат серии: для val/test — среднее и (при n > 1) несмещённое std
    по запускам метрик SUMMARY_METRICS, плюс best_val_top1 и n_params
    """
    out = {"n_runs": len(results), "n_params": results[0]["n_params"]}
    for split in ("val", "test"):
        agg = {}
        for metric in SUMMARY_METRICS:
            values = np.array([r[split][metric] for r in results if metric in r[split]])
            if len(values) == 0:
                continue
            agg[metric] = float(values.mean())
            if len(values) > 1:
                agg[f"{metric}_std"] = float(values.std(ddof=1))
        out[split] = agg
    out["best_val_top1"] = float(np.mean([r["best_val_top1"] for r in results]))
    return out
