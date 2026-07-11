"""
GEDLCE – Graph-Enhanced Dual Low-rank Correlation Embedding
Exact replication of the paper's formulation adapted for the MODMA dataset.

ADMM sub-problems:
  J  <- SVT  (low-rank of Z)
  Q  <- SVT  (nuclear-norm of P)
  E  <- shrinkage (sparse error)
  Z  <- linear solve (n×n  system)
  P  <- linear solve (d×d  system per column block) — NO Kronecker product needed
        because P is block-diagonal: each P_i is independent.
"""
import numpy as np
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


# --------------------------------------------------------------------------- #
# Proximal operators                                                            #
# --------------------------------------------------------------------------- #
def svt(X, tau):
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        U, S, Vt = la.svd(X, full_matrices=False, lapack_driver='gesvd')
    except Exception:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ np.diag(np.maximum(S - tau, 0)) @ Vt


def shrinkage(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


# --------------------------------------------------------------------------- #
# GEDLCE                                                                        #
# --------------------------------------------------------------------------- #
class GEDLCE:
    def __init__(self, n_views=2, p_dim=20,
                 lambda0=1.0, lambda1=0.1, lambda2=0.1, lambda3=0.1,
                 delta=1.0, max_iter=30, tol=1e-6, k_neighbors=10):
        self.n_views    = n_views
        self.p_dim      = p_dim
        self.lambda0    = lambda0
        self.lambda1    = lambda1
        self.lambda2    = lambda2
        self.lambda3    = lambda3
        self.delta      = delta
        self.max_iter   = max_iter
        self.tol        = tol
        self.k_neighbors = k_neighbors

    # ----------------------------------------------------------------------- #
    def fit(self, X_list, y):
        """
        X_list : list of m arrays, each (d_i, n_samples)
        y      : (n_samples,) integer labels
        """
        n_samples = X_list[0].shape[1]
        m  = self.n_views
        p  = self.p_dim
        mp = m * p

        d_list  = [X.shape[0] for X in X_list]
        d_total = sum(d_list)

        X = np.concatenate(X_list, axis=0)          # (d_total, n_samples)

        # ------------------------------------------------------------------ #
        # 1. Sparse k-NN graph Laplacians  (O(kn) instead of O(n²))
        # ------------------------------------------------------------------ #
        Xt = X.T                                     # (n_samples, d_total)
        k  = min(self.k_neighbors, n_samples - 1)
        A  = kneighbors_graph(Xt, n_neighbors=k,
                              mode='connectivity', include_self=False)
        A  = ((A + A.T) > 0).astype(float).toarray()  # symmetric binary

        same = (y[:, None] == y[None, :])
        W_w  = A *   same
        W_b  = A * (~same)
        L_w  = np.diag(W_w.sum(1)) - W_w            # (n, n)
        L_b  = np.diag(W_b.sum(1)) - W_b            # (n, n)

        # ------------------------------------------------------------------ #
        # 2. CCA cross-covariance C and L2,1 regulariser R (both d×d)
        # ------------------------------------------------------------------ #
        # C_{ij} = X_i @ X_j^T  (i≠j), zero on diagonal
        C_blocks = []
        for i in range(m):
            row = []
            for j in range(m):
                row.append(np.zeros((d_list[i], d_list[j])) if i == j
                           else X_list[i] @ X_list[j].T)
            C_blocks.append(row)
        C = np.block(C_blocks)                       # (d_total, d_total)

        # R_i = diag(1 / (2 * ‖row‖₂)) of X_i X_i^T
        R_diag = []
        for i in range(m):
            cov     = X_list[i] @ X_list[i].T
            rnorms  = la.norm(cov, axis=1)
            rnorms  = np.where(rnorms == 0, 1.0, rnorms)
            R_diag.append(np.diag(1.0 / (2.0 * rnorms)))
        R = la.block_diag(*R_diag)                   # (d_total, d_total)

        # Pre-compute fixed (d_total × d_total) matrices
        XLwXt = X @ L_w @ X.T
        XLbXt = X @ L_b @ X.T
        XXt   = X @ X.T

        # ------------------------------------------------------------------ #
        # 3. Initialise ADMM variables
        # ------------------------------------------------------------------ #
        J = np.zeros((n_samples, n_samples))
        Z = np.zeros((n_samples, n_samples))
        E = np.zeros((mp, n_samples))

        # P is block-diagonal: initialise each block with top-p PCA directions
        P_blocks = []
        for i in range(m):
            pca = PCA(n_components=p, random_state=42)
            pca.fit(X_list[i].T)
            P_blocks.append(pca.components_.T)       # (d_i, p)

        # P stored as list of blocks — avoids assembling the full (d×mp) matrix
        # We only need to track P_i for each view.

        # Multipliers (one per constraint per block)
        Y1_blocks = [np.zeros((p, n_samples)) for _ in range(m)]  # constraint: P_i^T X_i (I-Z) - E_i = 0
        Y3_blocks = [np.zeros((d_list[i], p)) for i in range(m)]  # constraint: P_i - Q_i = 0
        Q_blocks  = [P_blocks[i].copy() for i in range(m)]

        Y2 = np.zeros((n_samples, n_samples))        # constraint: Z - J = 0

        mu     = 1e-6
        max_mu = 1e6
        rho    = 1.2

        for iteration in range(self.max_iter):

            # ---------------------------------------------------------------- #
            # Compute PtX per view and concatenate for Z and E updates
            # PtX_i = P_i^T @ X_i  →  shape (p, n_samples)
            # ---------------------------------------------------------------- #
            PtX_blocks = [P_blocks[i].T @ X_list[i] for i in range(m)]
            PtX = np.concatenate(PtX_blocks, axis=0)   # (mp, n_samples)

            # ---- Update J -------------------------------------------------- #
            J = svt(Z + Y2 / mu, 1.0 / mu)

            # ---- Update Q (per view) --------------------------------------- #
            Q_blocks = [svt(P_blocks[i] + Y3_blocks[i] / mu,
                            self.lambda1 / mu)
                        for i in range(m)]

            # ---- Update E  ------------------------------------------------- #
            Psi = PtX - PtX @ Z + np.concatenate(Y1_blocks, axis=0) / mu
            E   = shrinkage(Psi, self.lambda0 / mu)

            # ---- Update Z  ------------------------------------------------- #
            lhs = PtX.T @ PtX + np.eye(n_samples)
            lhs = np.nan_to_num(lhs, nan=0.0, posinf=1e6, neginf=-1e6)
            lhs += 1e-8 * np.eye(n_samples)
            rhs = (PtX.T @ (PtX - E)
                   + J
                   + (PtX.T @ np.concatenate(Y1_blocks, axis=0) + Y2) / mu)
            rhs = np.nan_to_num(rhs, nan=0.0, posinf=1e6, neginf=-1e6)
            try:
                Z = la.solve(lhs, rhs)
            except la.LinAlgError:
                Z = la.lstsq(lhs, rhs)[0]

            # ---- Update each P_i independently (block-diagonal structure) -- #
            # Because P is block-diagonal, the Kronecker solve decomposes into
            # m independent  (d_i × d_i)  linear systems — one per view.
            # Gradient w.r.t. P_i of augmented Lagrangian gives:
            #   M_i @ P_i  =  RHS_i
            # M_i = 2λ2 A_i + mu * X_i X_i^T + mu I_di  -  2λ3 off_diag_i
            # where A_i = (XLwXt block for view i), off_diag_i from C block.
            start = 0
            E_blocks  = [E[i*p:(i+1)*p, :] for i in range(m)]
            PtXZ_blk  = [PtX_blocks[i] @ Z for i in range(m)]

            for i in range(m):
                di = d_list[i]
                Xi = X_list[i]                               # (di, n)

                # Diagonal block of XLwXt and XLbXt
                A_i = Xi @ L_w @ Xi.T                        # (di, di)

                # Off-diagonal cross-view correlation term from view j!=i.
                # d/dP_i tr(P^T C P) contributes sum_j C_ij P_j for block i.
                cross_i = sum(X_list[i] @ X_list[j].T @ P_blocks[j]
                              for j in range(m) if j != i)   # (di, p)

                # Within-view regulariser block
                cov_i   = Xi @ Xi.T
                rnorms_i = la.norm(cov_i, axis=1)
                rnorms_i = np.where(rnorms_i == 0, 1.0, rnorms_i)
                R_i = np.diag(1.0 / (2.0 * rnorms_i))       # (di, di)

                # System matrix  M_i  (di × di)
                M_i = (2 * self.lambda2 * A_i
                       + mu * Xi @ Xi.T
                       + mu * np.eye(di)
                       - 2 * self.lambda3 * self.delta * R_i)
                M_i = np.nan_to_num(M_i, nan=0.0, posinf=1e6, neginf=-1e6)
                M_i += 1e-8 * np.eye(di)

                # RHS_i  (di × p)
                RHS_i = (Xi @ Y1_blocks[i].T               # (di, p)
                         + Y3_blocks[i]                     # (di, p)
                         + 2 * self.lambda3 * cross_i
                         - mu * Xi @ (E_blocks[i] - PtX_blocks[i] + PtXZ_blk[i]).T)
                RHS_i = np.nan_to_num(RHS_i, nan=0.0, posinf=1e6, neginf=-1e6)

                try:
                    P_blocks[i] = la.solve(M_i, RHS_i)
                except la.LinAlgError:
                    # Fallback: gradient step
                    P_blocks[i] -= 0.01 * (M_i @ P_blocks[i] - RHS_i)

            # ---- Update multipliers --------------------------------------- #
            PtX_blocks = [P_blocks[i].T @ X_list[i] for i in range(m)]
            PtX        = np.concatenate(PtX_blocks, axis=0)

            diff1_blocks = [PtX_blocks[i] - PtX_blocks[i] @ Z - E_blocks[i]
                            for i in range(m)]
            diff2 = Z - J
            diff3_blocks = [P_blocks[i] - Q_blocks[i] for i in range(m)]

            for i in range(m):
                Y1_blocks[i] += mu * diff1_blocks[i]
                Y3_blocks[i] += mu * diff3_blocks[i]
            Y2 += mu * diff2
            mu  = min(rho * mu, max_mu)

            # ---- Convergence ---------------------------------------------- #
            errs = ([la.norm(d, 'fro') for d in diff1_blocks]
                  + [la.norm(diff2, 'fro')]
                  + [la.norm(d, 'fro') for d in diff3_blocks])
            if max(errs) < self.tol:
                break

        self.P_list = P_blocks       # list of (d_i, p) matrices
        return self

    # ----------------------------------------------------------------------- #
    def transform(self, X_list):
        """Returns (mp, n_samples) fused representation."""
        return np.concatenate(
            [self.P_list[i].T @ X_list[i] for i in range(self.n_views)],
            axis=0)
