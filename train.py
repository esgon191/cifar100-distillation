"""
Обучение модели на CIFAR-100 (учитель ResNet-50 или CNN-ученик) с логированием
в ClearML. Пример:
    python train.py --arch resnet50 --seed 0 --tags teacher
    python train.py --arch cnn --cnn_width 32 --seed 1 --tags student-ce
"""
import json
import os
import time

import torch

from cli.parser import get_parser
from cli.validators import validate_args
from kd.data import DeviceLoader, load_cifar100, stratified_split
from kd.models import build_model, count_params
from kd.trainer import (
    CrossEntropyObjective,
    build_optimizer,
    build_scheduler,
    evaluate,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
from utils.clearml_utils import get_clearml_dataset, report_single_values, report_split_metrics


def init_task(args):
    """
    Создаёт (или продолжает, если задан --continue_task_id) задачу ClearML.
    Артефакты и модели — на files server ClearML: адрес задаётся явно, т.к.
    output_uri=True подхватил бы default_output_uri (S3) из ~/clearml.conf.
    Автологирование torch.save отключено, чтобы промежуточные чекпоинты
    не расходовали хранилище
    """
    from clearml import Task
    from clearml.backend_api.session import Session

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name or f"{args.arch}-seed{args.seed}",
        task_type=Task.TaskTypes.training,
        tags=[args.arch, *args.tags],
        output_uri=Session.get_files_server_host(),
        continue_last_task=args.continue_task_id or False,
        auto_connect_frameworks={"pytorch": False},
    )
    task.connect(args, name="Args")
    return task


def find_checkpoint(ckpt_path: str, task) -> str | None:
    """
    Чекпоинт для продолжения: локальный файл запуска, иначе
    артефакт 'checkpoint' текущей задачи ClearML
    """
    if os.path.isfile(ckpt_path):
        return ckpt_path
    if task is not None and "checkpoint" in task.artifacts:
        return task.artifacts["checkpoint"].get_local_copy()
    return None


def main():
    args = get_parser().parse_args()
    validate_args(args)

    task = init_task(args) if args.clearml else None

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    amp = args.fp16 and device.type == "cuda"
    channels_last = device.type == "cuda"
    print(f"Устройство: {device}, AMP: {amp}")

    # --- Данные ---
    data_dir = args.data_dir
    if args.dataset_name:
        data_dir = get_clearml_dataset(args.dataset_name, args.dataset_version, args.project_name)
    data = load_cifar100(data_dir)

    train_idx, val_idx = stratified_split(data.y_train, args.val_size, args.split_seed)
    if args.max_train_samples is not None:
        train_idx = train_idx[:args.max_train_samples]
    print(f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(data.y_test)}")

    train_loader = DeviceLoader(
        data.x_train[train_idx], data.y_train[train_idx],
        batch_size=args.per_device_train_batch_size, device=device,
        shuffle=True, augment=True, drop_last=True, seed=args.seed,
        channels_last=channels_last
    )
    eval_kwargs = dict(
        batch_size=args.per_device_eval_batch_size, device=device,
        channels_last=channels_last
    )
    val_loader = DeviceLoader(data.x_train[val_idx], data.y_train[val_idx], **eval_kwargs)
    test_loader = DeviceLoader(data.x_test, data.y_test, **eval_kwargs)

    # --- Модель и оптимизация ---
    model = build_model(args).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"{args.arch}: {n_params / 1e6:.2f}M параметров")

    steps_per_epoch = len(train_loader)
    optimizer = build_optimizer(
        model, args.learning_rate, args.weight_decay, args.momentum, args.nesterov
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=args.num_train_epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    objective = CrossEntropyObjective(args.label_smoothing)

    # --- Продолжение с чекпоинта ---
    run_name = task.id if task is not None else (args.task_name or f"{args.arch}-seed{args.seed}")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")

    start_epoch, best_val_top1 = 0, 0.0
    if args.resume:
        found = find_checkpoint(ckpt_path, task)
        if found is not None:
            start_epoch, extra = load_checkpoint(found, model, optimizer, scheduler, scaler, device)
            best_val_top1 = extra.get("best_val_top1", 0.0)
            print(f"Продолжение с эпохи {start_epoch} ({found})")

    # --- Обучение ---
    val_metrics = {}
    for epoch in range(start_epoch, args.num_train_epochs):
        last_epoch = epoch == args.num_train_epochs - 1
        train_loader.set_epoch(epoch)
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, objective, device, amp
        )
        train_metrics["lr"] = scheduler.get_last_lr()[0]
        train_metrics["epoch_time"] = time.time() - t0

        log_line = (
            f"[{epoch + 1}/{args.num_train_epochs}] "
            f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.4f} "
            f"({train_metrics['epoch_time']:.1f}s)"
        )
        if task is not None:
            report_split_metrics("train", train_metrics, epoch + 1)

        if (epoch + 1) % args.eval_every == 0 or last_epoch:
            val_metrics = evaluate(model, val_loader, objective, device, amp)
            best_val_top1 = max(best_val_top1, val_metrics["top1"])
            log_line += f" | val loss {val_metrics['loss']:.4f} top1 {val_metrics['top1']:.4f}"
            if task is not None:
                report_split_metrics("val", val_metrics, epoch + 1)
        print(log_line)

        if (epoch + 1) % args.save_every == 0 or last_epoch:
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler, epoch,
                extra={"best_val_top1": best_val_top1}
            )
            if task is not None and args.upload_checkpoint:
                task.upload_artifact("checkpoint", artifact_object=ckpt_path, wait_on_upload=True)

    # --- Итоговые метрики: финальная модель (конец косинусного графика) ---
    if not val_metrics:
        val_metrics = evaluate(model, val_loader, objective, device, amp)
    test_metrics = evaluate(model, test_loader, objective, device, amp)
    summary = {
        "arch": args.arch,
        "seed": args.seed,
        "n_params": n_params,
        "val": val_metrics,
        "best_val_top1": best_val_top1,
        "test": test_metrics,
        "args": vars(args),
    }
    print(f"test top1 {test_metrics['top1']:.4f} top5 {test_metrics['top5']:.4f}")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    weights_path = os.path.join(run_dir, "model_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, weights_path)

    if task is not None:
        from clearml import OutputModel

        report_single_values("val", val_metrics)
        report_single_values("test", test_metrics)
        report_single_values("model", {"n_params": n_params, "best_val_top1": best_val_top1})
        task.upload_artifact("summary", artifact_object=summary)
        OutputModel(task=task, name=f"{args.arch}-seed{args.seed}", framework="PyTorch").update_weights(
            weights_filename=weights_path, auto_delete_file=False
        )
        task.close()


if __name__ == "__main__":
    main()
