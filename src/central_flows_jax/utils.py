import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.experimental.sparse.linalg import lobpcg_standard
from jax.tree_util import tree_flatten, tree_unflatten
from jaxtyping import Array

from .update_rules import Preconditioner


def mat_to_upper(x):
    k = x.shape[0]
    i, j = jnp.triu_indices(k)
    factors = jnp.where(i == j, 1.0, jnp.sqrt(2))
    return x[i, j] * factors


def _upper_to_mat(x):
    k = int(math.sqrt(2 * x.size))
    i, j = jnp.triu_indices(k)
    scale = jnp.where(i == j, 1.0, 1.0 / jnp.sqrt(2))
    vals = x * scale
    m = jnp.zeros((k, k), dtype=vals.dtype)
    m = m.at[i, j].set(vals)
    return m.at[j, i].set(vals)


def upper_to_mat(x, axes=None):
    if axes is None:
        axes = range(x.ndim)
    elif isinstance(axes, int):
        axes = (axes,)
    for axis in sorted([a % x.ndim for a in axes], reverse=True):
        x = jnp.apply_along_axis(_upper_to_mat, axis, x)
    return x


def apply_to_pairs(f, U: Array):
    k = U.shape[1]
    i, j = jnp.triu_indices(k)
    return lax.map(lambda c: f(*c), (U.T[i], U.T[j]))


def diff(f, x, order=1, *vs):
    assert len(vs) <= order
    if order == 0:
        return f(x)
    elif len(vs) == order:
        v, *vs = vs
        Df = lambda x: jax.jvp(f, [x], [v])[1]
    else:
        Df = jax.jacobian(f)
    return diff(Df, x, order - 1, *vs)


def compute_eigs(
    loss_fn: Callable,
    w: Array,
    refU: Array,
    P: Optional[Preconditioner] = None,
    tol: float = 1e-5,
) -> Tuple[Array, Array]:
    P_sqrt = (lambda x: x) if P is None else P.pow(1 / 2)
    P_inv_sqrt = (lambda x: x) if P is None else P.pow(-1 / 2)
    refV = vmap(P_sqrt, 1, 1)(refU)
    hvp = lambda v: P_inv_sqrt(diff(loss_fn, w, 2, P_inv_sqrt(v)))
    eff_tol = tol / (10 * len(w))  # undo lobpcg tol scaling
    evals, V, _ = lobpcg_standard(vmap(hvp, 1, 1), refV, tol=eff_tol)
    U = vmap(P_inv_sqrt, 1, 1)(V)
    return evals, U
