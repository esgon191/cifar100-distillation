"""
Smoke-тесты каркаса: разбиение, аугментации, модели, короткое обучение
с продолжением с чекпоинта. Запуск: python -m pytest tests
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from cli.parser import get_parser
from kd.data import DeviceLoader, load_cifar100, random_crop_flip, stratified_split
from kd.experiment import CrossEntropyExperiment, aggregate_runs, init_seeds
from kd.models import build_model, count_params, model_spec
from kd.trainer import build_optimizer, build_scheduler, macro_precision_recall, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "cifar_100")
has_data = os.path.isfile(os.path.join(DATA_DIR, "train"))


def test_stratified_split():
    labels = np.repeat(np.arange(100), 500)
    train_idx, val_idx = stratified_split(labels, 5000, split_seed=42)

    assert len(val_idx) == 5000 and len(train_idx) == 45000
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert (np.bincount(labels[val_idx]) == 50).all()

    again, _ = stratified_split(labels, 5000, split_seed=42)
    other, _ = stratified_split(labels, 5000, split_seed=7)
    assert (again == train_idx).all()
    assert not (other == train_idx).all()


def test_stratified_split_uneven():
    labels = np.repeat(np.arange(100), 500)
    _, val_idx = stratified_split(labels, 1234, split_seed=0)
    counts = np.bincount(labels[val_idx], minlength=100)
    assert len(val_idx) == 1234
    assert counts.min() >= 12 and counts.max() <= 13


def test_random_crop_flip():
    x = torch.randint(0, 256, (16, 3, 32, 32), dtype=torch.uint8)
    out1 = random_crop_flip(x, torch.Generator().manual_seed(0))
    out2 = random_crop_flip(x, torch.Generator().manual_seed(0))
    assert out1.shape == x.shape and out1.dtype == torch.uint8
    assert torch.equal(out1, out2)

    # Без паддинга и с принудительным отсутствием сдвига — либо тождество, либо отражение
    out = random_crop_flip(x, torch.Generator().manual_seed(1), padding=0)
    for a, b in zip(out, x):
        assert torch.equal(a, b) or torch.equal(a, b.flip(2))


def test_device_loader_epochs():
    x = np.random.randint(0, 256, (100, 3, 32, 32), dtype=np.uint8)
    y = np.arange(100)
    loader = DeviceLoader(x, y, batch_size=32, device=torch.device("cpu"),
                          shuffle=True, augment=True, drop_last=True, seed=3)
    assert len(loader) == 3

    loader.set_epoch(5)
    first = [yb.clone() for _, yb in loader]
    loader.set_epoch(5)
    second = [yb.clone() for _, yb in loader]
    assert all(torch.equal(a, b) for a, b in zip(first, second))

    xb, _ = next(iter(loader))
    assert xb.dtype == torch.float32 and xb.shape == (32, 3, 32, 32)


@pytest.mark.parametrize("arch", ["resnet50", "cnn"])
def test_models_forward(arch):
    args = get_parser().parse_args(["--arch", arch])
    model = build_model(args).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 100)
    n = count_params(model)
    if arch == "resnet50":
        assert 23e6 < n < 24e6
    else:
        assert n < 1e6


@pytest.mark.parametrize("decay_norm_bias", [True, False])
def test_build_optimizer_weight_decay(decay_norm_bias):
    model = build_model(get_parser().parse_args(["--arch", "resnet50"]))
    opt = build_optimizer(model, 0.1, 5e-4, decay_norm_bias=decay_norm_bias)
    n_total = sum(1 for _ in model.parameters())
    groups = opt.param_groups
    assert sum(len(g["params"]) for g in groups) == n_total

    if decay_norm_bias:
        assert len(groups) == 1 and groups[0]["weight_decay"] == 5e-4
    else:
        decay, no_decay = groups
        assert decay["weight_decay"] == 5e-4 and no_decay["weight_decay"] == 0.0
        assert all(p.ndim > 1 for p in decay["params"])
        assert all(p.ndim <= 1 for p in no_decay["params"])
        # BN-веса/сдвиги (2 на каждый BN) и bias классификатора
        n_bn = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
        assert len(no_decay["params"]) == 2 * n_bn + 1


def test_macro_precision_recall():
    # классы 0..3; класс 3 есть в метках, но ни разу не предсказан
    targets = torch.tensor([0, 0, 1, 1, 2, 3])
    preds = torch.tensor([0, 1, 1, 1, 2, 2])
    precision, recall = macro_precision_recall(preds, targets, num_classes=5)
    # precision: 0 -> 1/1, 1 -> 2/3, 2 -> 1/2, 3 -> 0 (нет предсказаний); класс 4 не учитывается
    assert precision == pytest.approx((1 + 2 / 3 + 0.5 + 0) / 4)
    # recall: 0 -> 1/2, 1 -> 2/2, 2 -> 1/1, 3 -> 0/1
    assert recall == pytest.approx((0.5 + 1 + 1 + 0) / 4)
    # на сбалансированной выборке macro recall совпадает с accuracy
    targets = torch.arange(4).repeat(5)
    preds = torch.randint(0, 4, (20,), generator=torch.Generator().manual_seed(0))
    _, recall = macro_precision_recall(preds, targets, num_classes=4)
    assert recall == pytest.approx((preds == targets).float().mean().item())


def test_init_seeds():
    seeds = init_seeds(10, 0)
    assert len(set(seeds)) == 10
    assert init_seeds(10, 0) == seeds          # детерминированы
    assert init_seeds(20, 0)[:10] == seeds     # расширение серии не меняет первые запуски
    assert init_seeds(10, 1) != seeds


def _experiment(seed: int = 0) -> CrossEntropyExperiment:
    args = get_parser().parse_args(["--arch", "cnn", "--cnn_width", "8", "--seed", str(seed)])
    model_cls, model_kwargs = model_spec(args)
    return CrossEntropyExperiment(
        args, model_cls, model_kwargs, loaders=None, device=torch.device("cpu"), run_dir=""
    )


def test_experiment_model_seed():
    exp = _experiment()
    state = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()])
    a, b, c = exp.build_model(1), exp.build_model(1), exp.build_model(2)
    assert torch.equal(state(a), state(b))      # тот же сид — те же веса
    assert not torch.equal(state(a), state(c))  # другой сид — другие веса

    # После создания модели глобальный генератор — от глобального сида, а не от сида весов
    exp.build_model(1)
    after_1 = torch.rand(3)
    exp.build_model(2)
    after_2 = torch.rand(3)
    set_seed(0)
    assert torch.equal(after_1, after_2) and torch.equal(after_1, torch.rand(3))


def test_aggregate_runs():
    runs = [
        {"n_params": 10, "best_val_top1": b, "val": {"top1": v}, "test": {"top1": t}}
        for b, v, t in [(0.5, 0.4, 0.3), (0.7, 0.6, 0.5)]
    ]
    agg = aggregate_runs(runs)
    assert agg["n_runs"] == 2
    assert agg["test"]["top1"] == pytest.approx(0.4)
    assert agg["test"]["top1_std"] == pytest.approx(np.std([0.3, 0.5], ddof=1))
    assert agg["best_val_top1"] == pytest.approx(0.6)
    assert "top1_std" not in aggregate_runs(runs[:1])["test"]


def test_scheduler_warmup_cosine():
    model = torch.nn.Linear(2, 2)
    opt = build_optimizer(model, 0.1, 0.0)
    sched = build_scheduler(opt, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.01)
    assert lrs[9] == pytest.approx(0.1)
    assert max(lrs) == pytest.approx(0.1)
    assert lrs[-1] < 1e-3


@pytest.mark.skipif(not has_data, reason="нет data/cifar_100")
def test_load_cifar100():
    data = load_cifar100(DATA_DIR, download=False)
    assert data.x_train.shape == (50000, 3, 32, 32)
    assert data.x_test.shape == (10000, 3, 32, 32)
    assert len(data.class_names) == 100
    assert (np.bincount(data.y_train) == 500).all()


@pytest.mark.skipif(not has_data, reason="нет data/cifar_100")
def test_train_resume(tmp_path):
    common = [
        sys.executable, os.path.join(ROOT, "train.py"),
        "--no-clearml", "--arch", "cnn", "--cnn_width", "8", "--cnn_convs_per_stage", "1",
        "--data_dir", DATA_DIR, "--output_dir", str(tmp_path),
        "--max_train_samples", "512", "--val_size", "500",
        "--per_device_train_batch_size", "64", "--save_every", "1",
        "--warmup_epochs", "0", "--task_name", "smoke",
        "--no-weight_decay_norm_bias",
    ]
    run = lambda epochs: subprocess.run(
        common + ["--num_train_epochs", str(epochs)],
        cwd=ROOT, capture_output=True, text=True, check=True
    )

    run(1)
    # Запуск 1 эпохи завершён (summary.json) — имитируем обрыв: оставляем только чекпоинт
    os.remove(tmp_path / "smoke" / "run_0" / "summary.json")
    out = run(2).stdout
    assert "Продолжение с эпохи 1" in out

    with open(tmp_path / "smoke" / "summary.json") as f:
        summary = json.load(f)
    assert summary["n_runs"] == 1
    assert 0 <= summary["test"]["top1"] <= 1
    assert os.path.isfile(tmp_path / "smoke" / "run_0" / "model_final.pt")


@pytest.mark.skipif(not has_data, reason="нет data/cifar_100")
def test_train_series(tmp_path):
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"),
        "--no-clearml", "--arch", "cnn", "--cnn_width", "8", "--cnn_convs_per_stage", "1",
        "--data_dir", DATA_DIR, "--output_dir", str(tmp_path),
        "--max_train_samples", "256", "--val_size", "500",
        "--per_device_train_batch_size", "64", "--warmup_epochs", "0",
        "--num_train_epochs", "1", "--n_runs", "2", "--task_name", "series",
    ]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    with open(tmp_path / "series" / "summary.json") as f:
        summary = json.load(f)
    assert summary["n_runs"] == 2
    seeds = [r["model_seed"] for r in summary["runs"]]
    assert seeds == init_seeds(2, 0)
    assert "top1_std" in summary["test"]

    # Повторный вызов: завершённые запуски пропускаются
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.count("уже завершён") == 2
