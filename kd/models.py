"""
Архитектуры: учитель ResNet-50 под 32×32 и небольшая CNN-ученик
"""
from argparse import Namespace
from collections.abc import Callable

import torch.nn as nn
from torchvision.models import resnet50


def resnet50_cifar(num_classes: int = 100) -> nn.Module:
    """
    ResNet-50 (torchvision), адаптированный под 32×32:
    стем conv3x3 stride 1 вместо conv7x7 stride 2, без maxpool.
    Последний BN каждого residual-блока инициализирован нулём
    """
    model = resnet50(weights=None, num_classes=num_classes, zero_init_residual=True)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model


def _conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


class SimpleCNN(nn.Module):
    """
    Небольшая CNN: 3 стадии по convs_per_stage блоков conv3x3-BN-ReLU
    шириной (width, 2·width, 4·width), между стадиями maxpool 2×2,
    затем global average pooling и линейный классификатор.
    Ёмкость регулируется width и convs_per_stage
    """

    def __init__(
            self,
            num_classes: int = 100,
            width: int = 32,
            convs_per_stage: int = 2
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for stage in range(3):
            out_ch = width * 2 ** stage
            for _ in range(convs_per_stage):
                layers += _conv_bn_relu(in_ch, out_ch)
                in_ch = out_ch
            if stage < 2:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


ARCHS = ("resnet50", "cnn")


def model_spec(args: Namespace) -> tuple[Callable[..., nn.Module], dict]:
    """
    Класс (фабрика) модели и аргументы конструктора по аргументам CLI
    (--arch и параметры ёмкости). Модель создаёт эксперимент — так он
    управляет сидом инициализации весов
    """
    if args.arch == "resnet50":
        return resnet50_cifar, {"num_classes": args.num_labels}
    if args.arch == "cnn":
        return SimpleCNN, {
            "num_classes": args.num_labels,
            "width": args.cnn_width,
            "convs_per_stage": args.cnn_convs_per_stage,
        }
    raise ValueError(f"Неизвестная архитектура: {args.arch}")


def build_model(args: Namespace) -> nn.Module:
    """
    Собирает модель по аргументам CLI (--arch и параметры ёмкости)
    """
    model_cls, model_kwargs = model_spec(args)
    return model_cls(**model_kwargs)


def count_params(model: nn.Module) -> int:
    """Число обучаемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
