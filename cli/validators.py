from argparse import ArgumentTypeError

def validate_args(args) -> None:
    """
    Валидатор для параметров передаваемых через командную строку
    """
    if args.learning_rate <= 0:
        raise ArgumentTypeError("--learning_rate должен быть > 0")
    if args.num_train_epochs <= 0:
        raise ArgumentTypeError("--num_train_epochs должен быть > 0")
    if not 0 <= args.warmup_epochs < args.num_train_epochs:
        raise ArgumentTypeError("--warmup_epochs должен быть в интервале [0, num_train_epochs)")
    if not 0 < args.val_size < 50000:
        raise ArgumentTypeError("--val_size должен быть в интервале (0, 50000)")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ArgumentTypeError("--max_train_samples должен быть > 0")
    if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
        raise ArgumentTypeError("размеры батчей должны быть > 0")
    if args.cnn_width <= 0 or args.cnn_convs_per_stage <= 0:
        raise ArgumentTypeError("--cnn_width и --cnn_convs_per_stage должны быть > 0")
    if not 0 <= args.label_smoothing < 1:
        raise ArgumentTypeError("--label_smoothing должен быть в интервале [0, 1)")
    if args.eval_every <= 0 or args.save_every <= 0:
        raise ArgumentTypeError("--eval_every и --save_every должны быть > 0")
    if args.n_runs <= 0:
        raise ArgumentTypeError("--n_runs должен быть > 0")
    if args.continue_task_id and not args.clearml:
        raise ArgumentTypeError("--continue_task_id требует --clearml")
