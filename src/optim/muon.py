from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, overload

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(update: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the zeroth power / polar factor of a matrix.

    This is the Newton-Schulz orthogonalization step used by Muon-like optimizers.
    The function expects a 2D tensor. If rows > columns, it transposes internally
    for numerical convenience and transposes back at the end.
    """
    if update.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(update.shape)}")

    original_dtype = update.dtype
    x = update.float()

    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    else:
        transposed = False

    x = x / (x.norm() + eps)

    # Coefficients used by common Muon implementations.
    # They are intentionally not the classical exact inverse-square-root coefficients;
    # this polynomial quickly pushes singular values toward 1 in a few iterations.
    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + b * (xx_t @ x) + c * (xx_t @ xx_t @ x)

    if transposed:
        x = x.T

    return x.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    """Single-process educational Muon optimizer for 2D parameters.

    This implementation is intentionally simple:
    - applies Muon only to 2D tensors;
    - keeps one momentum buffer per parameter;
    - supports decoupled weight decay;
    - does not implement distributed all-gather / sharding optimizations.
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 1e-5,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_steps <= 0:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                if param.ndim != 2:
                    raise ValueError(
                        f"Muon optimizer received a non-2D parameter. shape={tuple(param.shape)}"
                    )

                grad = param.grad

                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                state = self.state[param]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(momentum).add_(grad)

                update = grad.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Shape-dependent scaling used by many practical Muon variants.
                # It prevents very wide/tall matrices from receiving too small updates.
                update_scale = math.sqrt(max(1.0, param.size(0) / max(1, param.size(1))))
                update = update * update_scale

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                param.add_(update, alpha=-lr)

        return loss


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Wrapper that combines Muon for matrix params and AdamW for auxiliary params."""

    def __init__(
        self,
        muon_params: Sequence[torch.Tensor],
        adamw_params: Sequence[torch.Tensor],
        lr: float = 1e-5,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        if not muon_params:
            raise ValueError("muon_params is empty.")
        if not adamw_params:
            raise ValueError("adamw_params is empty.")

        self.muon = Muon(
            muon_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            foreach=False,
        )

        param_groups = self.muon.param_groups + self.adamw.param_groups
        defaults: dict[str, Any] = {}
        super().__init__(param_groups, defaults)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None

        if closure is not None:
            loss = closure()

        self.muon.step()
        self.adamw.step()

        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
