from dataclasses import dataclass
from functools import lru_cache

import cvxpy as cp
import jax
import numpy as np


@lru_cache(maxsize=None)
def get_sdcp_problem(k):
    A = cp.Parameter((k, k), name="A", symmetric=True)
    L = cp.Parameter((k**2, k**2), name="L")  # B = LLᵀ
    X = cp.Variable((k, k), name="X", PSD=True)
    # quadratic program: min_X  1/2 B[X,X] + ⟨A,X⟩
    # equivalent to the SDCP when B is PSD
    objective = 0.5 * cp.sum_squares(L.T @ cp.vec(X)) + cp.sum(cp.multiply(A, X))
    return cp.Problem(cp.Minimize(objective))


def solve_sdcp(A, B):
    A, B = np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)
    k = A.shape[0]
    if k == 0:
        return np.zeros((0, 0))
    if np.linalg.eigvalsh(A).min() >= 0:
        return np.zeros_like(A)

    # normalize B to have std 1 for numerical stability
    # we will undo this at the end
    B_size = np.sqrt(np.mean(B**2) + 1e-8)
    B = B.reshape(k**2, k**2)
    B = B / B_size

    # safely construct L, which satisfies B = LLᵀ
    B = (B + B.T) / 2
    evals, evecs = np.linalg.eigh(B)
    evals = np.maximum(evals, 0)
    L = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

    # compile the SDCP of size k if it doesn't exist

    # pass the parameters to the SDCP and solve
    sdcp_problem = get_sdcp_problem(k)
    params = sdcp_problem.param_dict
    params["A"].value = A
    params["L"].value = L
    sdcp_problem.solve(
        solver="CVXOPT",
        abstol=1e-11,
    )
    X = sdcp_problem.var_dict["X"].value

    return X / B_size


def jax_solve_sdcp(A, B):
    return jax.pure_callback(solve_sdcp, jax.ShapeDtypeStruct(A.shape, A.dtype), A, B)
