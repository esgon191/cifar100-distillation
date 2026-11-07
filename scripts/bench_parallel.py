"""
Замер скорости обучения группы из K моделей (kd/stacked.py) — для выбора --parallel_runs.
Использует тот же train_one_epoch, что и train.py (AMP, GradScaler, общий батч),
на случайных данных: важна только скорость. Запуск в Colab на T4:
    python scripts/bench_parallel.py --cnn_width 32 --cnn_convs_per_stage 2 --ks 1 2 5 10 20

Для каждого K печатает время шага группы, пропускную способность
(моделей·шагов в секунду) и ускорение относительно последовательного
обучения (K=1), а также пиковую память GPU
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kd.data import DeviceLoader  # noqa: E402
from kd.models import SimpleCNN  # noqa: E402
from kd.stacked import StackedModel  # noqa: E402
from kd.trainer import (  # noqa: E402
    CrossEntropyObjective,
    build_optimizer,
    build_scheduler,
    get_device,
    train_one_epoch,
)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Скорость обучения группы моделей от K")
    p.add_argument("--cnn_width", type=int, default=32)
    p.add_argument("--cnn_convs_per_stage", type=int, default=2)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--steps", type=int, default=300, help="Шагов на замер")
    p.add_argument("--warmup_steps", type=int, default=30, help="Шагов прогрева (не замеряются)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--channels_last", type=int, nargs="+", default=[0, 1],
                   help="Формат входа: 0 — contiguous, 1 — channels_last")
    return p


def bench(k: int, args, device: torch.device, channels_last: bool) -> dict:
    """Время шага группы из k моделей на args.steps шагах"""
    amp = device.type == "cuda"
    models = [SimpleCNN(100, args.cnn_width, args.cnn_convs_per_stage).to(device) for _ in range(k)]
    model = StackedModel(models)
    optimizer = build_optimizer(model, 0.1, 5e-4)
    scheduler = build_scheduler(optimizer, total_steps=10 ** 6, warmup_steps=0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    objective = CrossEntropyObjective()

    def loader(n_steps: int) -> DeviceLoader:
        n = n_steps * args.batch_size
        x = np.random.randint(0, 256, (n, 3, 32, 32), dtype=np.uint8)
        y = np.random.randint(0, 100, n)
        return DeviceLoader(x, y, args.batch_size, device, shuffle=True, augment=True,
                            drop_last=True, channels_last=channels_last)

    run = lambda ld: train_one_epoch(model, ld, optimizer, scheduler, scaler, objective, device, amp)
    run(loader(args.warmup_steps))
    measured = loader(args.steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    run(measured)
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_time = (time.time() - t0) / args.steps
    memory = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
    return {"step_time": step_time, "throughput": k / step_time, "memory_gb": memory}


def main():
    args = get_parser().parse_args()
    device = get_device()
    print(f"Устройство: {device}; w{args.cnn_width}/c{args.cnn_convs_per_stage}, "
          f"батч {args.batch_size}, шагов {args.steps}")
    for channels_last in args.channels_last:
        print(f"\nвход channels_last={bool(channels_last)}")
        print(f"{'K':>4} {'шаг, мс':>9} {'моделей·шагов/с':>16} {'ускорение':>10} {'память, ГБ':>11}")
        base = None
        for k in args.ks:
            r = bench(k, args, device, bool(channels_last))
            if base is None:
                # Пропускная способность одной модели — по первому K в списке (обычно 1)
                base = r["throughput"] / k
            speedup = r["throughput"] / base
            print(f"{k:>4} {r['step_time'] * 1e3:>9.1f} {r['throughput']:>16.1f} "
                  f"{speedup:>9.2f}x {r['memory_gb']:>11.2f}")
            del r
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
