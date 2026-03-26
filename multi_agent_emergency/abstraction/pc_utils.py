"""
pc_utils.py  –  Policy-Composed transition matrix
==================================================
Compose a policy matrix π with a block-column transition matrix P.

This is a local copy for the multi-agent-emergency project, simplified
for the 1-D dense case used by the roundabout abstraction.
"""

from typing import Union
import numpy as np
from scipy import sparse

Matrix = Union[np.ndarray, sparse.spmatrix]


def Pc(P_flat, pol: Matrix):
    """
    Compose policy with a block-column transition.

    P_flat : ndarray (N, N*nu)  OR  object with .P_det attribute
    pol    : (N, nu) dense or sparse policy matrix

    Returns
    -------
    For ndarray: (N, N) collapsed transition matrix.
    For objects with P_det: a new object with collapsed P_det (N, N).
    """
    # normalize policy -> CSR (N, nu)
    if not sparse.issparse(pol):
        pol = sparse.csr_matrix(pol)
    N, nu = pol.shape

    # policy vector in Fortran order: [π(:,1); π(:,2); ...]  (N*nu,)
    v = np.asarray(pol.toarray()).ravel(order="F")

    # ---- 2D / wrapper path (if it has P_det) ----
    if hasattr(P_flat, "P_det"):
        P_full = P_flat.P_det
        n = P_flat.n if hasattr(P_flat, 'n') else N

        if sparse.issparse(P_full):
            Plarge = P_full @ sparse.diags(v)
            blocks = Plarge.shape[1] // n
            P_comp = sparse.csr_matrix((n, n))
            for i in range(blocks):
                P_comp += Plarge[:, i*n:(i+1)*n]
        else:
            P_full = np.asarray(P_full, float)
            Plarge = P_full * v
            blocks = Plarge.shape[1] // N
            P_comp = Plarge.reshape(N, N, blocks, order="F").sum(axis=2)

        # If the input has Pi (2D tensor), rebuild the wrapper
        if hasattr(P_flat, 'Pi') and hasattr(P_flat, 'l1'):
            try:
                from src.models.utils.tensor_transition_probability_2d import TransitionProbability2D
                l = (P_flat.l1, P_flat.l2)
                K0, K1 = P_flat.Pi[0], P_flat.Pi[1]
                return TransitionProbability2D(l=l, P_det=P_comp, Pi0=K0, Pi1=K1)
            except ImportError:
                pass
        return P_comp

    # ---- 1D dense path: return (N, N) ndarray ----
    P_full = np.asarray(P_flat, float)  # (N, N*nu)
    Plarge = P_full * v
    blocks = Plarge.shape[1] // N
    return Plarge.reshape(N, N, blocks, order="F").sum(axis=2)
