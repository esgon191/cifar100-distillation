"""
Smoke-тесты каркаса: разбиение, аугментации, модели, группа моделей (vmap),
короткое обучение серий с продолжением с чекпоинта. Запуск: python -m pytest tests
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from cli.parser import get_parser
from kd.data import DeviceLoader, load_cifar100, random_crop_flip, stratified_split
from kd.experiment import (
    RUN_FILES,
    CrossEntropyExperiment,
    Loaders,
    aggregate_final,
    collect_series,
    init_seeds,
    missing_series_files,
    verify_series_files,
)
from kd.models import SimpleCNN, build_model, count_params, model_spec
from kd.stacked import StackedModel
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


def test_aggregate_final():
    final = pd.DataFrame([
        {"run": r, "model_seed": r, "split": s, "metric": "top1", "value": v}
        for r, s, v in [(0, "val", 0.4), (1, "val", 0.6), (0, "test", 0.3), (1, "test", 0.5)]
    ])
    agg = aggregate_final(final)
    assert agg["test"]["top1"] == pytest.approx(0.4)
    assert agg["test"]["top1_std"] == pytest.approx(np.std([0.3, 0.5], ddof=1))
    assert agg["val"]["top1"] == pytest.approx(0.5)
    assert "top1_std" not in aggregate_final(final[final.run == 0])["test"]


# --- Группа моделей (kd/stacked.py) ---

def _cnns(k: int, width: int = 4) -> list[torch.nn.Module]:
    models = []
    for seed in range(k):
        torch.manual_seed(seed)
        models.append(SimpleCNN(num_classes=10, width=width, convs_per_stage=1))
    return models


def test_stacked_matches_separate_models():
    """K моделей группы == K отдельных моделей: выходы, градиенты, BN-буферы, шаги SGD"""
    models = _cnns(3)
    refs = [copy.deepcopy(m) for m in models]
    stacked = StackedModel(models)
    opt = build_optimizer(stacked, 0.1, 5e-4, decay_norm_bias=False)
    ref_opts = [build_optimizer(m, 0.1, 5e-4, decay_norm_bias=False) for m in refs]

    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        x = torch.randn(8, 3, 32, 32, generator=gen)
        y = torch.randint(0, 10, (8,), generator=gen)
        out = stacked(x)
        assert out.shape == (3, 8, 10)
        loss = torch.stack([F.cross_entropy(out[k], y) for k in range(3)])
        opt.zero_grad()
        loss.sum().backward()
        for k, (m, o) in enumerate(zip(refs, ref_opts)):
            o.zero_grad()
            ref_out = m(x)
            assert torch.allclose(ref_out, out[k], atol=1e-5)
            F.cross_entropy(ref_out, y).backward()
            o.step()
        opt.step()

    for k, m in enumerate(refs):
        state = stacked.unstack(k)
        for name, value in m.state_dict().items():
            assert torch.allclose(value.float(), state[name].float(), atol=1e-5), name

    # eval: BN на running-статистиках каждой модели
    stacked.eval()
    x = torch.randn(4, 3, 32, 32, generator=gen)
    with torch.no_grad():
        out = stacked(x)
        for k, m in enumerate(refs):
            assert torch.allclose(m.eval()(x), out[k], atol=1e-5)


def test_stacked_unstack_and_single():
    models = _cnns(2)
    stacked = StackedModel(models).eval()
    x = torch.randn(4, 3, 32, 32)
    for k in range(2):
        plain = SimpleCNN(num_classes=10, width=4, convs_per_stage=1).eval()
        plain.load_state_dict(stacked.unstack(k))
        with torch.no_grad():
            assert torch.allclose(plain(x), stacked(x)[k], atol=1e-5)

    # K=1 — без vmap, тот же результат; градиент доходит до сложенных параметров
    single = StackedModel(_cnns(1))
    ref = _cnns(1)[0]
    out = single(x)
    assert out.shape == (1, 4, 10)
    assert torch.allclose(out[0], ref(x), atol=1e-5)
    out.sum().backward()
    assert all(p.grad is not None for p in single.parameters())
    # BN-буферы обновились в сложенных тензорах так же, как у обычной модели в train
    state = single.unstack(0)
    for name, value in ref.state_dict().items():
        assert torch.allclose(value.float(), state[name].float(), atol=1e-6), name


def test_stacked_optimizer_groups():
    """decay_norm_bias=False делит сложенные параметры так же, как параметры одной модели"""
    single = _cnns(1)[0]
    stacked = StackedModel(_cnns(3))
    n_single = [len(g["params"]) for g in build_optimizer(single, 0.1, 5e-4, decay_norm_bias=False).param_groups]
    n_stacked = [len(g["params"]) for g in build_optimizer(stacked, 0.1, 5e-4, decay_norm_bias=False).param_groups]
    assert n_single == n_stacked


# --- Серия запусков на синтетических данных ---

def _synthetic_loaders(n: int = 96) -> Loaders:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (n, 3, 32, 32), dtype=np.uint8)
    y = rng.integers(0, 10, n)
    cpu = torch.device("cpu")
    ev = dict(batch_size=40, device=cpu)
    return Loaders(
        train=DeviceLoader(x, y, batch_size=32, device=cpu, shuffle=True, augment=True,
                           drop_last=True, seed=0),
        val=DeviceLoader(x[:40], y[:40], **ev),
        test=DeviceLoader(x[40:], y[40:], **ev),
        train_eval=DeviceLoader(x, y, **ev),
    )


def _series(run_dir, parallel_runs: int, epochs: int = 2) -> CrossEntropyExperiment:
    args = get_parser().parse_args([
        "--arch", "cnn", "--cnn_width", "4", "--cnn_convs_per_stage", "1", "--num_labels", "10",
        "--num_train_epochs", str(epochs), "--warmup_epochs", "0", "--save_every", "1",
        "--parallel_runs", str(parallel_runs), "--task_name", "t",
    ])
    model_cls, model_kwargs = model_spec(args)
    set_seed(args.seed)
    return CrossEntropyExperiment(args, model_cls, model_kwargs, _synthetic_loaders(),
                                  torch.device("cpu"), str(run_dir))


def test_series_parallel_equals_sequential(tmp_path):
    """Результат запуска не зависит от того, обучался он один или в группе"""
    seq = _series(tmp_path / "seq", parallel_runs=1).run(3, init_seed=0)
    par = _series(tmp_path / "par", parallel_runs=2).run(3, init_seed=0)   # группы [0, 1], [2]
    for a, b in zip(seq, par):
        assert a["model_seed"] == b["model_seed"]
        for split in ("train", "val", "test"):
            for metric in ("loss", "ce", "top1", "top5"):
                assert a[split][metric] == pytest.approx(b[split][metric], abs=1e-4), (split, metric)
    for i in range(3):
        wa = torch.load(tmp_path / "seq" / f"run_{i}" / "model_final.pt")["model"]
        wb = torch.load(tmp_path / "par" / f"run_{i}" / "model_final.pt")["model"]
        for name in wa:
            assert torch.allclose(wa[name].float(), wb[name].float(), atol=1e-4), name


def test_series_files_and_resume(tmp_path, monkeypatch, capsys):
    exp = _series(tmp_path, parallel_runs=2, epochs=1)
    # Обрыв после обучения группы [0, 1], до финального замера: остаётся только чекпоинт
    original = CrossEntropyExperiment._final_measure

    def crash(self, *a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(CrossEntropyExperiment, "_final_measure", crash)
    with pytest.raises(KeyboardInterrupt):
        exp.run(3, init_seed=0)
    assert os.path.isfile(tmp_path / "group_0" / "checkpoint.pt")
    assert not os.path.isfile(tmp_path / "run_0" / "final.json")

    # Продолжение с parallel_runs=3: группа [0, 1] восстанавливается в том же составе
    monkeypatch.setattr(CrossEntropyExperiment, "_final_measure", original)
    exp = _series(tmp_path, parallel_runs=3, epochs=2)
    finals = exp.run(3, init_seed=0)
    out = capsys.readouterr().out
    assert "[runs [0, 1]] Продолжение с эпохи 1" in out
    assert [f["model_seed"] for f in finals] == init_seeds(3, 0)

    # История продолженного запуска — без повторов эпох
    hist = pd.read_csv(tmp_path / "run_0" / "history.csv")
    assert sorted(hist[(hist.split == "train") & (hist.metric == "loss")].epoch) == [1, 2]

    # Все файлы запусков на месте, чекпоинты групп удалены
    assert missing_series_files(str(tmp_path), 3, with_tables=False) == []
    assert not any(name.startswith("group_") for name in os.listdir(tmp_path))
    with open(tmp_path / "run_2" / "final.json") as f:
        final = json.load(f)
    assert set(final) >= {"run", "model_seed", "n_params", "train", "val", "test"}
    assert set(final["test"]) >= {"loss", "ce", "top1", "top5", "precision", "recall"}
    logits = np.load(tmp_path / "run_2" / "logits.npz")
    assert logits["test"].shape == (56, 10) and logits["test"].dtype == np.float16
    assert (logits["y_test"] == exp.loaders.test.y.numpy()).all()

    # Сводные таблицы и проверка файлов
    with pytest.raises(FileNotFoundError):
        verify_series_files(str(tmp_path), 3)
    tables = collect_series(str(tmp_path), 3)
    verify_series_files(str(tmp_path), 3)
    assert list(tables["test_metrics"].run) == [0, 1, 2]
    assert set(tables["final_metrics"].split) == {"train", "val", "test"}
    assert tables["history"].epoch.max() == 2

    # Повторный вызов: все запуски завершены — пропуск
    _series(tmp_path, parallel_runs=2, epochs=2).run(3, init_seed=0)
    assert capsys.readouterr().out.count("уже завершён") == 3


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
def test_train_series(tmp_path):
    """Сквозной запуск train.py: серия из 3 запусков группами по 2 на CIFAR-100"""
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"),
        "--no-clearml", "--arch", "cnn", "--cnn_width", "8", "--cnn_convs_per_stage", "1",
        "--data_dir", DATA_DIR, "--output_dir", str(tmp_path),
        "--max_train_samples", "256", "--val_size", "500",
        "--per_device_train_batch_size", "64", "--warmup_epochs", "0",
        "--num_train_epochs", "1", "--n_runs", "3", "--parallel_runs", "2",
        "--task_name", "series",
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    run_dir = tmp_path / "series"
    assert f"Каталог серии: {run_dir}" in out
    verify_series_files(str(run_dir), 3)
    with open(run_dir / "series.json") as f:
        series = json.load(f)
    assert series["model_seeds"] == init_seeds(3, 0) and series["parallel_runs"] == 2
    test = pd.read_csv(run_dir / "test_metrics.csv")
    assert len(test) == 3 and test.top1.between(0, 1).all()
    for i in range(3):
        for name in RUN_FILES:
            assert os.path.isfile(run_dir / f"run_{i}" / name)

    # Повторный вызов: завершённые запуски пропускаются; --report_only не обучает
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.count("уже завершён") == 3
    out = subprocess.run(cmd + ["--report_only"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    assert "уже завершён" not in out and "test top1" in out
