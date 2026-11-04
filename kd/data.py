"""
Загрузка CIFAR-100, фиксированное разбиение train/val и подача батчей
с аугментациями прямо на устройстве (весь датасет лежит в памяти GPU)
"""
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Статистики train CIFAR-100 для нормализации
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

CIFAR100_FILES = ("train", "test", "meta")


@dataclass
class Cifar100:
    """
    CIFAR-100 в виде numpy-массивов:
    изображения (N, 3, 32, 32) uint8, метки (N,) int64
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


def _find_cifar_dir(data_dir: str) -> str | None:
    """
    Ищет pickle-файлы CIFAR-100 (train, test, meta) в data_dir
    или в data_dir/cifar-100-python (раскладка torchvision)
    """
    for d in (data_dir, os.path.join(data_dir, "cifar-100-python")):
        if all(os.path.isfile(os.path.join(d, f)) for f in CIFAR100_FILES):
            return d
    return None


def _read_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _unpack(batch: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(batch[b"fine_labels"], dtype=np.int64)
    return x, y


def load_cifar100(data_dir: str, download: bool = True) -> Cifar100:
    """
    Читает CIFAR-100 (python-версия, pickle) из data_dir.
    Если файлов нет и download=True — скачивает через torchvision в data_dir
    """
    cifar_dir = _find_cifar_dir(data_dir)
    if cifar_dir is None:
        if not download:
            raise FileNotFoundError(f"CIFAR-100 не найден в {data_dir}")
        from torchvision.datasets import CIFAR100
        CIFAR100(root=data_dir, train=True, download=True)
        cifar_dir = _find_cifar_dir(data_dir)

    x_train, y_train = _unpack(_read_pickle(os.path.join(cifar_dir, "train")))
    x_test, y_test = _unpack(_read_pickle(os.path.join(cifar_dir, "test")))
    meta = _read_pickle(os.path.join(cifar_dir, "meta"))
    class_names = [name.decode() for name in meta[b"fine_label_names"]]

    return Cifar100(x_train, y_train, x_test, y_test, class_names)


def stratified_split(
        labels: np.ndarray,
        val_size: int,
        split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Стратифицированное разбиение индексов на train/val.
    Зависит только от split_seed (не от сида обучения), поэтому val
    одинаковый во всех запусках. Возвращает отсортированные индексы
    """
    rng = np.random.default_rng(split_seed)
    classes, counts = np.unique(labels, return_counts=True)

    # Квота val на класс пропорционально его размеру; остаток от округления
    # раздаётся классам с наибольшей дробной частью (ничьи — случайно)
    quota = val_size * counts / counts.sum()
    n_val = np.floor(quota).astype(np.int64)
    rest = val_size - int(n_val.sum())
    if rest > 0:
        perm = rng.permutation(len(classes))
        order = perm[np.argsort(-(quota - n_val)[perm], kind="stable")]
        n_val[order[:rest]] += 1

    val_parts = []
    for c, k in zip(classes, n_val):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        val_parts.append(idx[:k])

    val_idx = np.sort(np.concatenate(val_parts))
    train_mask = np.ones(len(labels), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)

    return train_idx, val_idx


def random_crop_flip(
        x: torch.Tensor,
        generator: torch.Generator,
        padding: int = 4
) -> torch.Tensor:
    """
    Батчевая аугментация на устройстве: RandomCrop(32, padding) с нулевым
    паддингом + RandomHorizontalFlip(0.5). Вход и выход — (B, C, H, W).
    Случайные числа берутся из CPU-генератора — воспроизводимо на любом устройстве
    """
    b, _, h, w = x.shape
    padded = F.pad(x, (padding, padding, padding, padding))

    dy = torch.randint(0, 2 * padding + 1, (b,), generator=generator).to(x.device)
    dx = torch.randint(0, 2 * padding + 1, (b,), generator=generator).to(x.device)
    flip = (torch.rand(b, generator=generator) < 0.5).to(x.device)

    rows = dy[:, None] + torch.arange(h, device=x.device)
    cols = dx[:, None] + torch.arange(w, device=x.device)
    # Отражение по горизонтали — просто обратный порядок столбцов
    cols = torch.where(flip[:, None], cols.flip(1), cols)

    batch = torch.arange(b, device=x.device)[:, None, None]
    out = padded[batch, :, rows[:, :, None], cols[:, None, :]]  # (B, H, W, C)

    return out.permute(0, 3, 1, 2)


class DeviceLoader:
    """
    Итератор по батчам (x, y): данные целиком лежат на устройстве в uint8,
    перемешивание, аугментации и нормализация делаются там же.
    Порядок и аугментации эпохи определяются (seed, epoch) — см. set_epoch,
    что делает продолжение с чекпоинта воспроизводимым
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            device: torch.device,
            shuffle: bool = False,
            augment: bool = False,
            drop_last: bool = False,
            seed: int = 0,
            channels_last: bool = False
    ):
        self.x = torch.as_tensor(x).to(device)
        self.y = torch.as_tensor(y).to(device)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.augment = augment
        self.drop_last = drop_last
        self.seed = seed
        self.channels_last = channels_last
        self.mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)
        self.generator = torch.Generator()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Переинициализирует генератор под эпоху"""
        self.generator.manual_seed(self.seed * 1_000_003 + epoch)

    def __len__(self) -> int:
        n = len(self.y)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.y)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator).to(self.device)
        else:
            order = torch.arange(n, device=self.device)

        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            xb = self.x[idx]
            if self.augment:
                xb = random_crop_flip(xb, self.generator)
            xb = (xb.float().div_(255) - self.mean) / self.std
            memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
            yield xb.contiguous(memory_format=memory_format), self.y[idx]
