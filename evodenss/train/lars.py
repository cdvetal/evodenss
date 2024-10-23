import math
from typing import Any, no_type_check

import torch
from torch import optim
from torch.nn.parameter import Parameter


class LARS(optim.Optimizer):

    def __init__(self,
                 params: Any,
                 batch_size: int,
                 lr_weights: float,
                 lr_biases: float,
                 weight_decay: float,
                 momentum: float,
                 eta: float=0.001,
                 weight_decay_filter: bool=False,
                 lars_adaptation_filter: bool=False) -> None:
        # we initialise lt with 0 just to comply with superclass params
        # in practice lt is going to be dynamic and separated
        # (one for weights and another for biases)
        defaults: dict[str, Any] = dict(
            lr=0.0,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
        super().__init__(params, defaults)
        self.batch_size: int = batch_size
        self.lr_weights: float = lr_weights
        self.lr_biases: float = lr_biases

    def adjust_learning_rate(self, n_batches: int, total_epochs: int, step: int) -> None:
        max_steps = total_epochs * n_batches
        warmup_steps = 10 * n_batches
        base_lr = self.batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        self.param_groups[0]['lr'] = lr * self.lr_weights
        self.param_groups[1]['lr'] = lr * self.lr_biases

    def exclude_bias_and_norm(self, p: Parameter) -> bool:
        return p.ndim == 1

    @torch.no_grad()
    @no_type_check
    def step(self, _=None) -> None:
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
