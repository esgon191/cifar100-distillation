"""
Учитель для KD и кэш его логитов.

Батчи и аугментации эпохи задаются только (--seed, epoch) (DeviceLoader.set_epoch),
поэтому логиты учителя на аугментированных батчах одинаковы для всех запусков
серии, всех KD-серий и всей сетки (α, T): они не зависят ни от ученика, ни от
параметров лосса. Их считаем один раз и храним на диске — это те же «онлайн»
логиты на тех же батчах, только посчитанные заранее.

Каталог кэша: <teacher_cache_dir>/<ключ>/, ключ — хеш параметров, от которых
зависят батчи и логиты (учитель, сид, размер батча, разбиение, AMP, устройство):
- meta.json — эти параметры и test top1 учителя (проверка, что веса те);
- train_e<epoch>.npz — логиты [steps, B, C] и метки [steps, B] эпохи;
- eval.npz — логиты и метки val, test, train_eval (без аугментаций).
Логиты хранятся в dtype выхода учителя (fp16 под AMP, иначе fp32) — кэш совпадает
с онлайн-проходом бит в бит. Файлы пишутся атомарно: готовая эпоха = файл есть,
построение продолжается после обрыва
"""
import hashlib
import json
import os
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

from kd.models import resnet50_cifar

EVAL_SPLITS = ("val", "test", "train_eval")


# --- Учитель ---

def load_teacher(
        device: torch.device,
        task_id: str | None = None,
        checkpoint: str | None = None,
        num_classes: int = 100,
        channels_last: bool = False
) -> nn.Module:
    """
    ResNet-50 учителя: веса — выходная модель задачи ClearML task_id
    (скачивается с files server в локальный кэш ClearML) или файл checkpoint.
    model_final.pt — {"model": state_dict, ...} или сам state_dict.
    Учитель в режиме eval, без градиентов
    """
    if (task_id is None) == (checkpoint is None):
        raise ValueError("Нужен ровно один источник учителя: task_id или checkpoint")
    if task_id is not None:
        from utils.clearml_utils import download_task_model
        checkpoint = download_task_model(task_id)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    teacher = resnet50_cifar(num_classes)
    teacher.load_state_dict(state)
    teacher = teacher.to(device)
    if channels_last:
        teacher = teacher.to(memory_format=torch.channels_last)
    teacher.eval().requires_grad_(False)
    return teacher


@torch.no_grad()
def teacher_logits(teacher: nn.Module, x: torch.Tensor, device: torch.device, amp: bool) -> torch.Tensor:
    """Логиты учителя на батче — так же, как в train_one_epoch (autocast при AMP)"""
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        return teacher(x)


# --- Кэш ---

def cache_identity(args: Namespace, n_train: int, steps_per_epoch: int,
                   device: torch.device, amp: bool) -> dict:
    """Параметры, от которых зависят батчи и логиты учителя: из них — ключ кэша"""
    teacher = args.teacher_task_id or os.path.abspath(args.teacher_checkpoint)
    return {
        "teacher": teacher,
        "seed": args.seed,
        "batch_size": args.per_device_train_batch_size,
        "split_seed": args.split_seed,
        "val_size": args.val_size,
        "max_train_samples": args.max_train_samples,
        "n_train": n_train,
        "steps_per_epoch": steps_per_epoch,
        "amp": amp,
        "device_type": device.type,
    }


class TeacherCache:
    """Логиты учителя на диске: эпохи train и eval-части"""

    def __init__(self, root: str, identity: dict):
        key = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = os.path.join(root, key)
        self.identity = identity
        os.makedirs(self.dir, exist_ok=True)
        meta = self.meta()
        if meta and meta["identity"] != identity:
            raise RuntimeError(f"{self.dir}: meta.json не совпадает с параметрами кэша")
        if not meta:
            self._write_meta({"identity": identity})

    # Файлы

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def epoch_path(self, epoch: int) -> str:
        return self._path(f"train_e{epoch:03d}.npz")

    def has_epoch(self, epoch: int) -> bool:
        return os.path.isfile(self.epoch_path(epoch))

    def has_eval(self) -> bool:
        return os.path.isfile(self._path("eval.npz"))

    def meta(self) -> dict:
        path = self._path("meta.json")
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _write_meta(self, meta: dict) -> None:
        _atomic(self._path("meta.json"), lambda f: f.write(json.dumps(meta, indent=2).encode()))

    def update_meta(self, **values) -> None:
        self._write_meta({**self.meta(), **values})

    def save_epoch(self, epoch: int, logits: np.ndarray, y: np.ndarray) -> None:
        _atomic(self.epoch_path(epoch), lambda f: np.savez(f, logits=logits, y=y))

    def save_eval(self, arrays: dict[str, np.ndarray]) -> None:
        _atomic(self._path("eval.npz"), lambda f: np.savez(f, **arrays))

    # Чтение

    def load_epoch(self, epoch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Логиты [steps, B, C] и метки [steps, B] эпохи — на устройство"""
        if not self.has_epoch(epoch):
            raise FileNotFoundError(f"В кэше учителя нет эпохи {epoch}: {self.epoch_path(epoch)}")
        with np.load(self.epoch_path(epoch)) as data:
            return (torch.from_numpy(data["logits"]).to(device),
                    torch.from_numpy(data["y"].astype(np.int64)).to(device))

    def load_eval(self, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Логиты [N, C] и метки [N] eval-части в порядке загрузчика"""
        with np.load(self._path("eval.npz")) as data:
            return (torch.from_numpy(data[split]).to(device),
                    torch.from_numpy(data[f"y_{split}"].astype(np.int64)).to(device))


def _atomic(path: str, write) -> None:
    """Запись через временный файл: при обрыве не остаётся «полу-файла»"""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


@torch.no_grad()
def build_teacher_cache(teacher: nn.Module, loaders, n_epochs: int, cache: TeacherCache,
                        device: torch.device, amp: bool) -> None:
    """
    Достраивает кэш: недостающие эпохи train (батчи — как в обучении: set_epoch(e),
    затем проход загрузчика) и eval-части. Учитель — в eval. Печатает прогресс
    и test top1 учителя (пишется в meta.json)
    """
    teacher.eval()
    missing = [e for e in range(n_epochs) if not cache.has_epoch(e)]
    if missing:
        print(f"Кэш учителя {cache.dir}: считаю эпох {len(missing)} из {n_epochs}")
    for epoch in missing:
        t0 = time.time()
        loaders.train.set_epoch(epoch)
        logits, ys = [], []
        for x, y in loaders.train:
            logits.append(teacher_logits(teacher, x, device, amp).cpu())
            ys.append(y.cpu())
        cache.save_epoch(epoch, torch.stack(logits).numpy(),
                         torch.stack(ys).numpy().astype(np.uint8))
        print(f"  эпоха {epoch + 1}/{n_epochs}: {time.time() - t0:.1f}s")

    if not cache.has_eval():
        arrays = {}
        for split in EVAL_SPLITS:
            loader = getattr(loaders, split)
            if loader is None:
                continue
            arrays[split] = torch.cat(
                [teacher_logits(teacher, x, device, amp).cpu() for x, _ in loader]
            ).numpy()
            arrays[f"y_{split}"] = loader.y.cpu().numpy().astype(np.uint8)
        cache.save_eval(arrays)
        top1 = float((arrays["test"].argmax(-1) == arrays["y_test"]).mean())
        cache.update_meta(teacher_test_top1=top1)
    print(f"Кэш учителя готов: {cache.dir}, test top1 учителя {cache.meta()['teacher_test_top1']:.4f}")


@torch.no_grad()
def verify_cache(teacher: nn.Module, train_loader, cache: TeacherCache, epoch: int,
                 device: torch.device, amp: bool, atol: float = 0.05) -> float:
    """
    Первый батч эпохи epoch: учитель онлайн против кэша. Ловит изменение
    аугментаций, порядка батчей или весов. Возвращает максимальное расхождение.
    atol — на случай разных алгоритмов cuDNN между прогонами (fp16)
    """
    train_loader.set_epoch(epoch)
    x, y = next(iter(train_loader))
    online = teacher_logits(teacher, x, device, amp).float()
    cached, cached_y = cache.load_epoch(epoch, device)
    diff = (online - cached[0].float()).abs().max().item()
    if not torch.equal(cached_y[0], y) or diff > atol:
        raise RuntimeError(
            f"Кэш учителя {cache.dir} не соответствует батчам эпохи {epoch}: "
            f"метки {'совпали' if torch.equal(cached_y[0], y) else 'не совпали'}, "
            f"макс. расхождение логитов {diff:.4f}"
        )
    train_loader.set_epoch(epoch)
    return diff


class TeacherLogitsLoader:
    """
    Загрузчик с логитами учителя из кэша: батчи (x, y, логиты учителя).
    split="train" — логиты эпохи грузятся в set_epoch; иначе — логиты eval-части.
    Метки каждого батча сверяются с кэшем (на устройстве, проверка — в конце прохода):
    расхождение значит, что кэш построен для других батчей
    """

    def __init__(self, loader, cache: TeacherCache, split: str, device: torch.device):
        self.loader = loader
        self.cache = cache
        self.split = split
        self.device = device
        self.y = loader.y
        self._epoch = None
        if split != "train":
            self._logits, self._y = cache.load_eval(split, device)

    def set_epoch(self, epoch: int) -> None:
        self.loader.set_epoch(epoch)
        if self.split == "train" and epoch != self._epoch:
            self._logits, self._y = self.cache.load_epoch(epoch, self.device)
            self._epoch = epoch

    def __len__(self) -> int:
        return len(self.loader)

    def _batch_logits(self, i: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            return self._logits[i], self._y[i]
        bs = self.loader.batch_size
        return self._logits[i * bs:i * bs + b], self._y[i * bs:i * bs + b]

    def __iter__(self):
        if self.split == "train" and self._epoch is None:
            raise RuntimeError("TeacherLogitsLoader(train): перед проходом нужен set_epoch")
        mismatch = torch.zeros((), dtype=torch.long, device=self.device)
        for i, (x, y) in enumerate(self.loader):
            logits, cached_y = self._batch_logits(i, y.shape[0])
            mismatch += (cached_y != y).sum()
            yield x, y, logits
        if mismatch.item():
            raise RuntimeError(
                f"Кэш учителя {self.cache.dir} ({self.split}): метки не совпали "
                f"с батчами загрузчика — кэш построен для других данных"
            )
