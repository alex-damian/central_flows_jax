from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array


class UpdateRule:
    def initialize_state(self, w: jnp.ndarray) -> Array:
        raise NotImplementedError()

    def P(self, flat_state: Array) -> Preconditioner:
        raise NotImplementedError()

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        raise NotImplementedError()

    def dstate_dt(self, flat_state: Array, gradient: Array):
        return self.update_state(flat_state, gradient) - flat_state


@dataclass
class GradientDescent(UpdateRule):
    lr: float

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array) -> Array:
        state = {"t": jnp.array(0.0, dtype=w.dtype)}
        flat_state, self.unflatten = ravel_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Preconditioner:
        state = self.unflatten(flat_state)
        return DiagonalPreconditioner(1 / self.lr_fn(state["t"]))

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        state = {"t": state["t"] + 1.0}
        return ravel_pytree(state)[0]


@dataclass
class RMSProp(UpdateRule):
    lr: float
    beta2: float
    eps: float = 0
    bias_correction: bool = False
    scalar: bool = False

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array) -> Array:
        zero = jnp.array(0.0, dtype=w.dtype)
        nu0 = zero if self.scalar else jnp.zeros_like(w)
        state = {"t": zero, "nu": nu0}
        flat_state, self.unflatten = ravel_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Preconditioner:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        if self.bias_correction:
            nu /= 1 - self.beta2**t
        lrs = self.lr_fn(t) / (jnp.sqrt(nu) + self.eps)
        return DiagonalPreconditioner(1 / lrs)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        g2 = jnp.square(gradient)
        if self.scalar:
            g2 = g2.sum()
        nu += (1 - self.beta2) * (g2 - nu)
        state = {"t": t + 1.0, "nu": nu}
        return ravel_pytree(state)[0]

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        update = self.update_state(flat_state, gradient) - flat_state
        return update / self.beta2


class Preconditioner:
    def __call__(self, v: Array) -> Array:
        raise NotImplementedError()

    def pow(self, p: float) -> Preconditioner:
        raise NotImplementedError()


@dataclass(frozen=True)
class DiagonalPreconditioner(Preconditioner):
    P: Any

    def __call__(self, v: Array) -> Array:
        return v * self.P

    def pow(self, p: float) -> DiagonalPreconditioner:
        return DiagonalPreconditioner(self.P**p)


def to_schedule(schedule_or_constant):
    if callable(schedule_or_constant):
        return schedule_or_constant
    else:
        return lambda t: schedule_or_constant
