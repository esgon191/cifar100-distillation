"""
Цикл обучения и оценки: SGD + warmup/cosine, AMP (fp16) на CUDA,
чекпоинты для продолжения обучения после обрыва сессии
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
    L2-штраф не регуляризует, а лишь сжимает масштаб
    """
    if decay_norm_bias:
        params = model.parameters()
    else:
        decay, no_decay = [], []
        for p in model.parameters():
            if p.requires_grad:
                (no_decay if p.ndim <= 1 else decay).append(p)
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


class CrossEntropyObjective:
    """
    Обычная перекрёстная энтропия по меткам.
    Интерфейс целевой функции: (logits, targets, teacher_logits) -> (loss, компоненты),
    компоненты — словарь detached-тензоров для логирования
    """

    def __init__(self, label_smoothing: float = 0.0):
        self.label_smoothing = label_smoothing

    def __call__(self, logits, targets, teacher_logits=None):
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return loss, {"ce": loss.detach()}


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    topk = logits.topk(k, dim=1).indices
    return (topk == targets[:, None]).any(dim=1).sum()


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
) -> dict:
    """
    Одна эпоха обучения. Если передан teacher — он прогоняется онлайн
    на тех же аугментированных батчах, его логиты уходят в objective.
    Возвращает средние по примерам loss, компоненты лосса и top1 на train
    """
    model.train()
    if teacher is not None:
        # Учитель всегда в режиме eval: BN на накопленных статистиках
        teacher.eval()
    sums: dict[str, torch.Tensor] = {}
    n_seen = 0

    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(x)
        loss, parts = objective(
            logits.float(),
            y,
            None if teacher_logits is None else teacher_logits.float()
        )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        b = y.shape[0]
        n_seen += b
        batch_stats = {"loss": loss.detach(), **parts}
        for name, value in batch_stats.items():
            sums[name] = sums.get(name, 0) + value.float() * b
        sums["top1"] = sums.get("top1", 0) + (logits.detach().argmax(1) == y).sum()

    metrics = {name: (value / n_seen).item() for name, value in sums.items()}
    if not math.isfinite(metrics["loss"]):
        raise FloatingPointError(f"Лосс на train не конечен: {metrics['loss']}")
    return metrics


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader,
        objective,
        device: torch.device,
        amp: bool,
        teacher: nn.Module | None = None
) -> dict:
    """
    Оценка без аугментаций: loss, компоненты лосса, top1, top5
    """
    model.eval()
    if teacher is not None:
        teacher.eval()
    sums: dict[str, torch.Tensor] = {}
    n_seen = 0

    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            teacher_logits = teacher(x) if teacher is not None else None
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
        sums["top1"] = sums.get("top1", 0) + (logits.argmax(1) == y).sum()
        sums["top5"] = sums.get("top5", 0) + _topk_correct(logits, y, 5)

    return {name: (value / n_seen).item() for name, value in sums.items()}


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
