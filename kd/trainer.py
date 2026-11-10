"""
Цикл обучения и оценки: SGD + warmup/cosine, AMP (fp16) на CUDA,
чекпоинты для продолжения обучения после обрыва сессии.

Цикл работает с группой из K моделей (kd/stacked.py::stack_models):
модель возвращает логиты [K, B, C], целевая функция — лосс каждой модели [K],
метрики возвращаются списком из K словарей. Одиночная модель — случай K=1
"""
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Фиксирует сиды random/numpy/torch. deterministic=True дополнительно
    включает детерминированные алгоритмы cuDNN (медленнее)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    """CUDA, если есть, иначе MPS, иначе CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float = 0.9,
        nesterov: bool = True,
        decay_norm_bias: bool = True
) -> torch.optim.Optimizer:
    """
    SGD с momentum. decay_norm_bias=False исключает из weight decay
    одномерные параметры (веса/сдвиги BatchNorm и bias'ы) — для них
    L2-штраф не регуляризует, а лишь сжимает масштаб. У сложенных параметров
    группы моделей размерность берётся у одной модели (_base_ndim)
    """
    if decay_norm_bias:
        params = model.parameters()
    else:
        decay, no_decay = [], []
        for p in model.parameters():
            if p.requires_grad:
                ndim = getattr(p, "_base_ndim", p.ndim)
                (no_decay if ndim <= 1 else decay).append(p)
        params = [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    return torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov
    )


def build_scheduler(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Пошаговый (по итерациям) график LR: линейный warmup, затем косинус до 0
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def per_model_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.0
) -> torch.Tensor:
    """CE каждой модели группы: logits [K, B, C], targets [B] -> [K]"""
    k, b, c = logits.shape
    ce = F.cross_entropy(
        logits.reshape(k * b, c), targets.repeat(k),
        reduction="none", label_smoothing=label_smoothing
    )
    return ce.view(k, b).mean(1)


class CrossEntropyObjective:
    """
    Обычная перекрёстная энтропия по меткам.
    Интерфейс целевой функции группы моделей:
    (logits [K, B, C], targets [B], teacher_logits [B, C] | None) -> (loss [K], компоненты),
    компоненты — словарь detached-тензоров [K] для логирования
    """

    def __init__(self, label_smoothing: float = 0.0):
        self.label_smoothing = label_smoothing

    def __call__(self, logits, targets, teacher_logits=None):
        loss = per_model_cross_entropy(logits, targets, self.label_smoothing)
        return loss, {"ce": loss.detach()}


def per_model_kl(logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    KL(softmax(z_t/T) ‖ softmax(z_s/T)) каждой модели группы, среднее по примерам:
    logits [K, B, C], teacher_logits [B, C] (общие для группы) -> [K]
    """
    log_ps = F.log_softmax(logits / temperature, dim=-1)
    log_pt = F.log_softmax(teacher_logits / temperature, dim=-1)
    return (log_pt.exp() * (log_pt - log_ps)).sum(-1).mean(-1)


class KDObjective:
    """
    Дистилляция: α·CE(y, z_s) + (1−α)·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)).
    Множитель T² выравнивает масштаб градиента KL-слагаемого (он ∝ 1/T²) с CE.
    Компоненты: ce и kl — kl без множителя T², как мера расхождения распределений
    ученика и учителя при температуре T
    """

    def __init__(self, alpha: float, temperature: float):
        self.alpha = alpha
        self.temperature = temperature

    def __call__(self, logits, targets, teacher_logits=None):
        if teacher_logits is None:
            raise ValueError("KDObjective: нужны логиты учителя (teacher или кэш логитов)")
        ce = per_model_cross_entropy(logits, targets)
        kl = per_model_kl(logits, teacher_logits, self.temperature)
        loss = self.alpha * ce + (1 - self.alpha) * self.temperature ** 2 * kl
        return loss, {"ce": ce.detach(), "kl": kl.detach()}


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Число попаданий в top-k для каждой модели: logits [K, B, C] -> [K]"""
    topk = logits.topk(k, dim=-1).indices
    return (topk == targets[None, :, None]).any(dim=-1).sum(dim=-1)


def _per_model(sums: dict[str, torch.Tensor], n_seen: int) -> list[dict]:
    """Суммы [K] по примерам -> список из K словарей средних"""
    means = {name: (value / n_seen).tolist() for name, value in sums.items()}
    k = len(next(iter(means.values())))
    return [{name: values[i] for name, values in means.items()} for i in range(k)]


def macro_precision_recall(
        preds: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
) -> tuple[float, float]:
    """
    Macro precision и macro recall: метрика считается по каждому классу,
    затем усредняется без весов. Усреднение — по классам, встречающимся
    в истинных метках; класс без предсказаний получает precision 0
    (как zero_division=0 в sklearn)
    """
    tp = torch.bincount(targets[preds == targets], minlength=num_classes).float()
    n_pred = torch.bincount(preds, minlength=num_classes).float()
    n_true = torch.bincount(targets, minlength=num_classes).float()
    present = n_true > 0
    precision = torch.where(n_pred > 0, tp / n_pred.clamp(min=1), torch.zeros_like(tp))
    recall = tp / n_true.clamp(min=1)
    return precision[present].mean().item(), recall[present].mean().item()


def _unpack(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Батч (x, y) или (x, y, логиты учителя из кэша) -> (x, y, логиты | None)"""
    if len(batch) == 3:
        return batch
    x, y = batch
    return x, y, None


def train_one_epoch(
        model: nn.Module,
        loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        objective,
        device: torch.device,
        amp: bool,
        teacher: nn.Module | None = None
) -> tuple[list[dict], int]:
    """
    Одна эпоха обучения группы моделей на общих батчах. Если передан teacher —
    он прогоняется онлайн один раз на батч, его логиты уходят в objective
    всех моделей группы. Для backward берётся сумма лоссов моделей: параметры
    не пересекаются, поэтому каждая модель получает градиент своего лосса.

    Возвращает (метрики каждой модели: средние по примерам loss, компоненты
    лосса и top1 на train; число шагов, пропущенных GradScaler из-за
    переполнения fp16 — общее на группу)
    """
    model.train()
    if teacher is not None:
        # Учитель всегда в режиме eval: BN на накопленных статистиках
        teacher.eval()
    sums: dict[str, torch.Tensor] = {}
    n_seen, skipped = 0, 0

    for batch in loader:
        x, y, teacher_logits = _unpack(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            if teacher_logits is None and teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(x)
        loss, parts = objective(
            logits.float(),
            y,
            None if teacher_logits is None else teacher_logits.float()
        )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss.sum()).backward()
        scaler.step(optimizer)
        if amp:
            # Шаг пропущен (inf/nan в fp16) — скейлер уменьшает масштаб
            scale = scaler.get_scale()
            scaler.update()
            skipped += int(scaler.get_scale() < scale)
        else:
            scaler.update()
        scheduler.step()

        b = y.shape[0]
        n_seen += b
        batch_stats = {"loss": loss.detach(), **parts}
        for name, value in batch_stats.items():
            sums[name] = sums.get(name, 0) + value.float() * b
        sums["top1"] = sums.get("top1", 0) + (logits.detach().argmax(-1) == y).sum(-1)

    metrics = _per_model(sums, n_seen)
    bad = [i for i, m in enumerate(metrics) if not math.isfinite(m["loss"])]
    if bad:
        raise FloatingPointError(f"Лосс на train не конечен у моделей группы {bad}")
    return metrics, skipped


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader,
        objective,
        device: torch.device,
        amp: bool,
        teacher: nn.Module | None = None,
        return_logits: bool = False
) -> list[dict] | tuple[list[dict], torch.Tensor]:
    """
    Оценка группы моделей без аугментаций. Для каждой модели: loss,
    компоненты лосса, top1, top5, macro precision и macro recall
    (по предсказаниям всей выборки). return_logits=True дополнительно
    возвращает логиты [K, N, C] (fp16, на CPU) в порядке загрузчика
    """
    model.eval()
    if teacher is not None:
        teacher.eval()
    sums: dict[str, torch.Tensor] = {}
    n_seen = 0
    all_preds, all_targets, all_logits = [], [], []
    num_classes = None

    for batch in loader:
        x, y, teacher_logits = _unpack(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            if teacher_logits is None and teacher is not None:
                teacher_logits = teacher(x)
        logits = logits.float()
        loss, parts = objective(
            logits,
            y,
            None if teacher_logits is None else teacher_logits.float()
        )

        b = y.shape[0]
        n_seen += b
        for name, value in {"loss": loss, **parts}.items():
            sums[name] = sums.get(name, 0) + value.float() * b
        preds = logits.argmax(-1)
        sums["top1"] = sums.get("top1", 0) + (preds == y).sum(-1)
        sums["top5"] = sums.get("top5", 0) + _topk_correct(logits, y, 5)
        all_preds.append(preds)
        all_targets.append(y)
        if return_logits:
            all_logits.append(logits.half().cpu())
        num_classes = logits.shape[-1]

    metrics = _per_model(sums, n_seen)
    preds, targets = torch.cat(all_preds, dim=1), torch.cat(all_targets)
    for k, m in enumerate(metrics):
        m["precision"], m["recall"] = macro_precision_recall(preds[k], targets, num_classes)
    if return_logits:
        return metrics, torch.cat(all_logits, dim=1)
    return metrics


def save_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        epoch: int,
        extra: dict | None = None
) -> None:
    """
    Сохраняет полное состояние обучения после эпохи epoch (0-based).
    Пишется во временный файл и атомарно переименовывается
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        device: torch.device
) -> tuple[int, dict]:
    """
    Восстанавливает состояние обучения. Возвращает (эпоха, с которой
    продолжать, extra)
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return state["epoch"] + 1, state["extra"]
