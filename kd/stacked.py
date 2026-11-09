"""
Группа из K моделей одной архитектуры как одна модель: параметры и буферы
K копий сложены по новой оси 0, прямой проход векторизован (torch.func.vmap).

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
