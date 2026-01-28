from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import grad, lax, vmap
from jax.experimental.sparse.linalg import lobpcg_standard
from jaxtyping import Array

from .sdcp import jax_solve_sdcp
from .update_rules import Preconditioner, UpdateRule
from .utils import apply_to_pairs, compute_eigs, diff, mat_to_upper, upper_to_mat


def central_flow_substep(
    loss_fn: Callable,
    opt: UpdateRule,
    w: Array,
    state: Array,
    refU: Array,
    dt: float,
) -> Tuple[Tuple[Array, Array, Array], dict]:
    L, g = jax.value_and_grad(loss_fn)(w)
    P = opt.P(state)
    eigs, U = compute_eigs(loss_fn, w, refU, P)
    k = U.shape[1]

    dH_dw_fn = lambda u, v: diff(loss_fn, w, 3, u, v)
    dH_dw = apply_to_pairs(dH_dw_fn, U)  #            NOTE: [binom(k+1,2), dim(w)]
    dw_dt_stable = -P.pow(-1)(g)  #                   NOTE: [dim(w)]
    dw_dt_X = -0.5 * vmap(P.pow(-1))(dH_dw).T  #      NOTE: [dim(w), binom(k+1,2)]
    HU = vmap(P, 1, 1)(U) * eigs[None, :]  #          NOTE: [dim(w), k]

    ds_dt_stable = opt.dstate_dt(state, g)  #         NOTE: [dim(state)]
    ds_dt_X_fn = lambda u, v: 0.5 * diff(lambda g: opt.dstate_dt(state, g), g, 2, u, v)
    ds_dt_X = apply_to_pairs(ds_dt_X_fn, U).T  #      NOTE: [dim(state), binom(k+1,2)]
    dH_dt_stable = dH_dw @ dw_dt_stable  #            NOTE: [binom(k+1,2)]
    dH_dt_X = dH_dw @ dw_dt_X  #                      NOTE: [binom(k+1,2), binom(k+1,2)]

    dP_ds_fn = lambda u, v: grad(lambda s: u @ opt.P(s)(v))(state)
    dP_ds = apply_to_pairs(dP_ds_fn, U)  #            NOTE: [binom(k+1,2), dim(state)]
    dP_dt_stable = dP_ds @ ds_dt_stable  #            NOTE: [binom(k+1,2)]
    dP_dt_X = dP_ds @ ds_dt_X  #                      NOTE: [binom(k+1,2), binom(k+1,2)]

    A = mat_to_upper(jnp.diag(2 - eigs))  #           NOTE: [binom(k+1,2)]
    dA_dt_stable = 2 * dP_dt_stable - dH_dt_stable  # NOTE: [binom(k+1,2)]
    dA_dt_X = 2 * dP_dt_X - dH_dt_X  #                NOTE: [binom(k+1,2), binom(k+1,2)]
    next_A_stable = A + dt * dA_dt_stable  #          NOTE: [binom(k+1,2)]
    next_A_X = dt * dA_dt_X  #                        NOTE: [binom(k+1,2), binom(k+1,2)]

    alpha = upper_to_mat(next_A_stable)  #            NOTE: [k,k]
    beta = upper_to_mat(next_A_X)  #                  NOTE: [k,k,k,k]
    X = jax_solve_sdcp(alpha, beta)  #                NOTE: [k,k]
    X_vec = mat_to_upper(X)  #                        NOTE: [binom(k+1,2)]

    dw_dt = dw_dt_stable + dw_dt_X @ X_vec  #         NOTE: [dim(w)]
    dstate_dt = ds_dt_stable + ds_dt_X @ X_vec  #     NOTE: [dim(state)]

    w += dt * dw_dt
    state += dt * dstate_dt
    return (w, state, U), dict(L=L, eigs=eigs, X=X)


@jax.tree_util.register_static
class Flow:
    def __init__(self, loss_fn: Callable, opt: UpdateRule):
        self.loss_fn = loss_fn
        self.opt = opt

    @jax.jit(static_argnames="midpoint")
    def discrete(self, w, state, refU, *, midpoint: bool = False):
        L, g = jax.value_and_grad(self.loss_fn)(w)
        new_state = self.opt.update_state(state, g)
        new_w = w - self.opt.P(new_state).pow(-1)(g)
        w_mid = w + 0.5 * (new_w - w)
        state_mid = state + 0.5 * (new_state - state)
        L_mid = self.loss_fn(w_mid)
        if midpoint:
            eigs, U = compute_eigs(self.loss_fn, w_mid, refU, self.opt.P(state_mid))
        else:
            eigs, U = compute_eigs(self.loss_fn, w, refU, self.opt.P(state))
        return (new_w, new_state, U), dict(L=L, L_mid=L, eigs=eigs)

    @jax.jit
    def stable(self, w, state, refU, eps=0.5):
        L = self.loss_fn(w)
        P = self.opt.P(state)
        eigs, U = compute_eigs(self.loss_fn, w, refU, P)
        n_substeps = jnp.ceil(eigs.max() / eps).astype(int)
        dt = 1 / n_substeps

        def substep(w, state):
            g = jax.grad(self.loss_fn)(w)
            P = self.opt.P(state)
            dstate_dt = self.opt.dstate_dt(state, g)
            dw_dt = -P.pow(-1)(g)
            new_state = state + dt * dstate_dt
            new_w = w + dt * dw_dt
            return new_w, new_state

        w, state = lax.fori_loop(
            0, n_substeps, lambda i, val: substep(*val), (w, state)
        )
        return (w, state, U), dict(L=L, eigs=eigs)

    @jax.jit
    def central(self, w, state, refU, substeps=4):
        dt = 1 / substeps

        def step_fn(val, _):
            w, state, refU = val
            return central_flow_substep(self.loss_fn, self.opt, w, state, refU, dt)

        (w, state, U), aux = lax.scan(step_fn, (w, state, refU), None, length=substeps)
        aux = jax.tree.map(lambda x: x[0], aux)
        return (w, state, U), aux

    @staticmethod
    @jax.jit(static_argnums=(0,))
    def make_pred(f, w, U, X):
        f0 = f(w)
        UtH = lax.map(lambda v: diff(f, w, 2, v), U.T)
        UtHU = UtH @ U
        return f0 + 0.5 * jnp.einsum("i...j,ij->...", UtHU, X)
