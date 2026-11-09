"""
Группа из K моделей одной архитектуры как одна модель, forward -> [K, B, C]:
- GroupedCNN — для SimpleCNN: K моделей как одна сеть на grouped conv.
  Быстрая реализация, поддерживает channels_last и AMP;
- StackedModel — для любой архитектуры: параметры и буферы K копий сложены по
  новой оси 0, прямой проход векторизован (torch.func.vmap). На T4 медленнее
  одной модели с channels_last, а channels_last vmap не поддерживает.
Выбор — stack_models.

Модели не делят ни параметров, ни буферов: SGD с momentum и weight decay
поэлементный, поэтому один оптимизатор на группу эквивалентен K независимым,
а градиент суммы лоссов по параметрам модели k равен градиенту её лосса.
BatchNorm в режиме train обновляет running-статистики каждой модели отдельно
"""
import copy
import types

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state


def _fp32_batch_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    BatchNorm на входе fp32, выход — в dtype входа.
    Правило vmap для batch_norm с собственными буферами у каждой модели передаёт
    в cudnn_batch_norm фиктивный вес ones(..., dtype входа). Под AMP вход fp16,
    а cuDNN требует вес fp32 — без приведения обучение падает с RuntimeError.
    cuDNN и так считает BN в fp32, поэтому результат почти не меняется
    """
    return type(self).forward(self, x.float()).to(x.dtype)


class StackedModel(nn.Module):
    """
    K моделей одной архитектуры: forward(x[B, ...]) -> logits[K, B, C].
    Все модели получают один и тот же батч
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("Нужна хотя бы одна модель")
        self.k = len(models)
        params, buffers = stack_module_state(models)

        # Имена исходной модели (с точками) не годятся для регистрации — храним списком
        self.param_names = list(params)
        self.buffer_names = list(buffers)
        self.stacked_params = nn.ParameterList()
        for value in params.values():
            p = nn.Parameter(value.detach().clone())
            # Размерность параметра одной модели: по ней build_optimizer отличает
            # BN/bias (ndim <= 1) от весов conv/linear
            p._base_ndim = p.ndim - 1
            self.stacked_params.append(p)
        for i, value in enumerate(buffers.values()):
            self.register_buffer(f"stacked_buffer_{i}", value.clone())

        # Каркас архитектуры без данных: не регистрируется как подмодуль,
        # чтобы его (пустые) параметры не попадали в parameters()/state_dict()
        base = copy.deepcopy(models[0]).to("meta")
        for m in base.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.forward = types.MethodType(_fp32_batch_norm_forward, m)
        object.__setattr__(self, "_base", base)
        self._base.train(models[0].training)
        self.train(models[0].training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Режим каркаса определяет поведение BN (batch- или running-статистики)
        self._base.train(mode)
        return self

    def _buffers_list(self) -> list[torch.Tensor]:
        return [getattr(self, f"stacked_buffer_{i}") for i in range(len(self.buffer_names))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = list(self.stacked_params)
        buffers = self._buffers_list()

        if self.k == 1:
            # Одна модель — без vmap: срезы [0] — представления, градиенты и
            # обновления BN-буферов попадают в сложенные тензоры
            state = {n: p[0] for n, p in zip(self.param_names, params)}
            state.update({n: b[0] for n, b in zip(self.buffer_names, buffers)})
            return functional_call(self._base, state, (x,)).unsqueeze(0)

        def single(p: dict, b: dict, inp: torch.Tensor) -> torch.Tensor:
            return functional_call(self._base, (p, b), (inp,))

        return torch.vmap(single, in_dims=(0, 0, None), randomness="same")(
            dict(zip(self.param_names, params)),
            dict(zip(self.buffer_names, buffers)),
            x
        )

    @torch.no_grad()
    def unstack(self, k: int) -> dict[str, torch.Tensor]:
        """state_dict модели k в формате исходной архитектуры (грузится в model_cls)"""
        state = {n: p[k].detach().clone() for n, p in zip(self.param_names, self.stacked_params)}
        state.update({n: b[k].clone() for n, b in zip(self.buffer_names, self._buffers_list())})
        return state


def _cat_into(dst: torch.Tensor, srcs: list[torch.Tensor]) -> None:
    """Записывает тензоры K моделей подряд по оси 0 (порядок каналов: модель, затем канал)"""
    dst.copy_(torch.cat([s.detach() for s in srcs], dim=0).reshape(dst.shape))


class GroupedCNN(nn.Module):
    """
    K моделей SimpleCNN как одна «широкая» сеть: forward(x[B, ...]) -> logits[K, B, C].
    Каналы модели k занимают k-й блок каналов каждого слоя:
    - первая свёртка — обычная 3 -> K·w (вход у всех моделей один);
    - остальные свёртки — groups=K, классификатор — grouped conv 1×1;
    - BatchNorm на K·C каналов: статистики по каналу, т.е. по каждой модели отдельно;
    - ReLU и пулинги поканальные.
    Это ровно K независимых моделей, но без vmap: запусков ядер столько же, сколько
    у одной модели, работают channels_last и AMP. Имена в state_dict совпадают с
    SimpleCNN, unstack(k) режет каждый тензор на K блоков по оси 0
    """

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("Нужна хотя бы одна модель")
        # Новые слои инициализируются случайно и сразу перезаписываются весами моделей;
        # глобальный генератор не расходуется, чтобы группа не меняла случайность серии
        with torch.random.fork_rng(devices=[]):
            self._build(models)

    def _build(self, models: list[nn.Module]) -> None:
        k = self.k = len(models)
        ref = models[0]
        layers: list[nn.Module] = []
        for i, layer in enumerate(ref.features):
            srcs = [m.features[i] for m in models]
            if isinstance(layer, nn.Conv2d):
                first = i == 0
                new = nn.Conv2d(
                    layer.in_channels * (1 if first else k), layer.out_channels * k,
                    kernel_size=layer.kernel_size, padding=layer.padding,
                    bias=layer.bias is not None, groups=1 if first else k
                )
            elif isinstance(layer, nn.BatchNorm2d):
                new = nn.BatchNorm2d(layer.num_features * k, eps=layer.eps, momentum=layer.momentum)
            else:
                new = copy.deepcopy(layer)
            with torch.no_grad():
                for name, t in list(new.named_parameters()) + list(new.named_buffers()):
                    parts = [getattr(s, name) for s in srcs]
                    if name == "num_batches_tracked":
                        t.copy_(parts[0])
                    else:
                        _cat_into(t, parts)
            layers.append(new)
        self.features = nn.Sequential(*layers)
        self.pool = copy.deepcopy(ref.pool)

        lin = ref.classifier
        self.num_classes = lin.out_features
        self.classifier = nn.Conv2d(lin.in_features * k, lin.out_features * k, 1,
                                    bias=lin.bias is not None, groups=k)
        with torch.no_grad():
            _cat_into(self.classifier.weight, [m.classifier.weight for m in models])
            if lin.bias is not None:
                _cat_into(self.classifier.bias, [m.classifier.bias for m in models])

        self._shapes = {n: t.shape for n, t in ref.state_dict().items()}
        self.to(next(ref.parameters()).device)
        self.train(ref.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        x = self.classifier(x).flatten(1)
        return x.view(x.shape[0], self.k, self.num_classes).transpose(0, 1)

    @torch.no_grad()
    def unstack(self, k: int) -> dict[str, torch.Tensor]:
        """state_dict модели k в формате SimpleCNN"""
        state = {}
        for name, t in self.state_dict().items():
            if name.endswith("num_batches_tracked"):
                state[name] = t.clone()
            else:
                part = t.reshape(self.k, -1)[k]
                state[name] = part.reshape(self._shapes[name]).clone()
        return state


def stack_models(models: list[nn.Module], channels_last: bool = False) -> nn.Module:
    """
    Группа из K моделей с интерфейсом forward -> [K, B, C] и unstack(k).
    SimpleCNN — GroupedCNN (grouped conv, быстро, поддерживает channels_last);
    прочие архитектуры — StackedModel (vmap). vmap не поддерживает channels_last,
    поэтому StackedModel с K > 1 и channels_last — ошибка
    """
    from kd.models import SimpleCNN

    if all(type(m) is SimpleCNN for m in models):
        group = GroupedCNN(models)
        return group.to(memory_format=torch.channels_last) if channels_last else group
    if len(models) > 1 and channels_last:
        raise ValueError(
            "Группа через vmap (StackedModel) не поддерживает channels_last: "
            "для этой архитектуры используйте --parallel_runs 1"
        )
    return StackedModel(models)
