"""
Интеграционные тесты KD на настоящем учителе: веса ResNet-50 скачиваются из задачи
ClearML (шаг 2), данные — CIFAR-100 из data/cifar_100. Нужны ~/clearml.conf (доступ
к app.clear.ml) и данные, иначе тесты пропускаются. Устройство — локальное
(на Mac — MPS, fp32). Запуск:
    python -m pytest tests -m teacher        # только они
    python -m pytest tests -m "not teacher"  # всё остальное, без сети
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from cli.parser import TEACHER_TASK_ID, get_parser
from kd.data import DeviceLoader, load_cifar100, stratified_split
from kd.experiment import KDExperiment, Loaders, verify_series_files
from kd.models import model_spec
from kd.teacher_cache import (
    TeacherCache,
    TeacherLogitsLoader,
    build_teacher_cache,
    cache_identity,
    load_teacher,
    teacher_logits,
    verify_cache,
)
from kd.trainer import get_device, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar_100")
TEACHER_TEST_TOP1 = 0.7951  # задача ClearML учителя, AMP на T4

pytestmark = [
    pytest.mark.teacher,
    pytest.mark.skipif(not os.path.isfile(os.path.join(DATA_DIR, "train")), reason="нет CIFAR-100"),
    pytest.mark.skipif(not os.path.isfile(os.path.expanduser("~/clearml.conf")),
                       reason="нет ~/clearml.conf"),
]

DEVICE = get_device()


@pytest.fixture(scope="module")
def data():
    return load_cifar100(DATA_DIR)


@pytest.fixture(scope="module")
def teacher():
    return load_teacher(DEVICE, task_id=TEACHER_TASK_ID)


def _args(*extra: str):
    return get_parser().parse_args([
        "--arch", "cnn", "--cnn_width", "8", "--cnn_convs_per_stage", "1",
        "--teacher_task_id", "--max_train_samples", "1024", "--num_train_epochs", "2",
        "--warmup_epochs", "0", "--save_every", "1", "--no-clearml", "--task_name", "kd",
        *extra,
    ])


def _loaders(data, args, eval_n: int | None = 512) -> Loaders:
    """Загрузчики как в train.py; eval-части урезаны до eval_n примеров ради скорости"""
    train_idx, val_idx = stratified_split(data.y_train, args.val_size, args.split_seed)
    train_idx = train_idx[:args.max_train_samples]
    val_idx = val_idx[:eval_n]
    test_n = eval_n or len(data.y_test)
    ev = dict(batch_size=args.per_device_eval_batch_size, device=DEVICE)
    return Loaders(
        train=DeviceLoader(data.x_train[train_idx], data.y_train[train_idx],
                           batch_size=args.per_device_train_batch_size, device=DEVICE,
                           shuffle=True, augment=True, drop_last=True, seed=args.seed),
        val=DeviceLoader(data.x_train[val_idx], data.y_train[val_idx], **ev),
        test=DeviceLoader(data.x_test[:test_n], data.y_test[:test_n], **ev),
        train_eval=DeviceLoader(data.x_train[train_idx], data.y_train[train_idx], **ev),
    )


def _cache(tmp_path, args, loaders) -> TeacherCache:
    identity = cache_identity(args, len(loaders.train.y), len(loaders.train), DEVICE, False)
    return TeacherCache(str(tmp_path / "cache"), identity)


# --- Получение учителя из хранилища ClearML ---

def test_download_teacher():
    """Веса скачиваются с files server; повторно — из локального кэша ClearML (тот же файл)"""
    from utils.clearml_utils import download_task_model

    first = download_task_model(TEACHER_TASK_ID)
    second = download_task_model(TEACHER_TASK_ID)
    assert first == second and os.path.isfile(first)
    state = torch.load(first, map_location="cpu", weights_only=False)
    assert set(state) >= {"model", "args"} and state["args"]["arch"] == "resnet50"


def test_teacher_quality(teacher, data):
    """Полный test: top1 восстановленного учителя совпадает с отчётом задачи ClearML"""
    loader = DeviceLoader(data.x_test, data.y_test, batch_size=500, device=DEVICE)
    correct = sum((teacher_logits(teacher, x, DEVICE, False).argmax(-1) == y).sum().item()
                  for x, y in loader)
    top1 = correct / len(data.y_test)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    assert top1 == pytest.approx(TEACHER_TEST_TOP1, abs=3e-3), top1


# --- Кэш логитов на настоящем учителе ---

def test_teacher_cache(tmp_path, teacher, data):
    args = _args()
    loaders = _loaders(data, args)
    cache = _cache(tmp_path, args, loaders)
    build_teacher_cache(teacher, loaders, 2, cache, DEVICE, False)
    assert cache.has_epoch(0) and cache.has_epoch(1) and cache.has_eval()
    assert 0.5 < cache.meta()["teacher_test_top1"] <= 1

    # Логиты кэша == учитель онлайн на тех же (аугментированных) батчах эпохи
    logits, ys = cache.load_epoch(1, DEVICE)
    assert logits.shape == (len(loaders.train), args.per_device_train_batch_size, 100)
    loaders.train.set_epoch(1)
    for i, (x, y) in enumerate(loaders.train):
        assert torch.equal(ys[i], y)
        assert torch.allclose(teacher_logits(teacher, x, DEVICE, False), logits[i], atol=1e-4)
    assert verify_cache(teacher, loaders.train, cache, 0, DEVICE, False) < 1e-4

    # Обёртка отдаёт (x, y, логиты) и для train, и для eval-частей
    wrapped = TeacherLogitsLoader(loaders.train, cache, "train", DEVICE)
    wrapped.set_epoch(1)
    batches = list(wrapped)
    assert len(batches) == len(loaders.train) and torch.equal(batches[0][2], logits[0])
    val = list(TeacherLogitsLoader(loaders.val, cache, "val", DEVICE))
    assert sum(b[2].shape[0] for b in val) == len(loaders.val.y)

    # Обрыв: файла эпохи 1 нет — достраивается только он
    mtime0 = os.path.getmtime(cache.epoch_path(0))
    os.remove(cache.epoch_path(1))
    build_teacher_cache(teacher, loaders, 2, cache, DEVICE, False)
    assert cache.has_epoch(1) and os.path.getmtime(cache.epoch_path(0)) == mtime0

    # Метки в кэше не те (кэш от других батчей) — ошибка сверки
    with np.load(cache.epoch_path(1)) as f:
        bad_logits, bad_y = f["logits"], f["y"][::-1].copy()
    cache.save_epoch(1, bad_logits, bad_y)
    wrapped = TeacherLogitsLoader(loaders.train, cache, "train", DEVICE)
    wrapped.set_epoch(1)
    with pytest.raises(RuntimeError, match="метки не совпали"):
        list(wrapped)

    # Другой сид — другие батчи — другой каталог кэша
    other = _cache(tmp_path, _args("--seed", "1"), loaders)
    assert other.dir != cache.dir


# --- Обучение: кэш == учитель онлайн ---

def _kd_series(tmp_path, data, name: str, *extra: str) -> list[dict]:
    args = _args("--n_runs", "2", "--parallel_runs", "2",
                 "--teacher_cache_dir", str(tmp_path / "cache"), *extra)
    set_seed(args.seed)
    model_cls, model_kwargs = model_spec(args)
    exp = KDExperiment(args, model_cls, model_kwargs, _loaders(data, args), DEVICE,
                       str(tmp_path / name))
    return exp.run(2, init_seed=0)


def test_kd_cache_equals_online(tmp_path, data):
    cached = _kd_series(tmp_path, data, "cached")
    online = _kd_series(tmp_path, data, "online", "--teacher_online")
    for a, b in zip(cached, online):
        for split in ("train", "val", "test"):
            assert a[split]["kl"] > 0
            for metric in ("loss", "ce", "kl", "top1", "top5"):
                assert a[split][metric] == pytest.approx(b[split][metric], abs=1e-4), (split, metric)
    for i in range(2):
        wa = torch.load(tmp_path / "cached" / f"run_{i}" / "model_final.pt")["model"]
        wb = torch.load(tmp_path / "online" / f"run_{i}" / "model_final.pt")["model"]
        for name in wa:
            assert torch.allclose(wa[name].float(), wb[name].float(), atol=1e-4), name


# --- Сквозной train.py ---

def test_train_kd_series(tmp_path):
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"),
        "--no-clearml", "--arch", "cnn", "--cnn_width", "8", "--cnn_convs_per_stage", "1",
        "--data_dir", DATA_DIR, "--output_dir", str(tmp_path),
        "--max_train_samples", "512", "--warmup_epochs", "0", "--num_train_epochs", "1",
        "--n_runs", "2", "--parallel_runs", "2", "--task_name", "kd-series",
        "--teacher_task_id", "--kd_alpha", "0.5", "--kd_temperature", "4",
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "KD: α=0.5, T=4.0" in out and "Кэш учителя готов" in out
    run_dir = tmp_path / "kd-series"
    verify_series_files(str(run_dir), 2)
    with open(run_dir / "series.json") as f:
        series = json.load(f)
    assert series["teacher_test_top1"] == pytest.approx(TEACHER_TEST_TOP1, abs=3e-3)
    assert series["teacher_cache"].startswith(str(tmp_path / "teacher_cache"))
    history = pd.read_csv(run_dir / "history.csv")
    assert {"ce", "kl"} <= set(history.metric)
    assert "kl" in pd.read_csv(run_dir / "test_metrics.csv").columns
