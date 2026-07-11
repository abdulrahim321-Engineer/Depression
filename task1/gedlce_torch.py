import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    torch = None

def svt_torch(X, tau):
    X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_thresh = torch.clamp(S - tau, min=0.0)
    return U @ torch.diag(S_thresh) @ Vh

def shrinkage_torch(X, tau):
    return torch.sign(X) * torch.clamp(torch.abs(X) - tau, min=0.0)

class GEDLCETorch:
    def __init__(self, n_views=2, p_dim=20,
                 lambda0=1.0, lambda1=0.1, lambda2=0.1, lambda3=0.1,
                 delta=1.0, max_iter=30, tol=1e-6, k_neighbors=10):
        self.n_views = n_views
        self.p_dim = p_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.k_neighbors = k_neighbors

    def fit(self, X_list_np, y):
        n_samples = X_list_np[0].shape[1]
        m = self.n_views
        p = self.p_dim
        mp = m * p

        d_list = [X.shape[0] for X in X_list_np]

        X_np = np.concatenate(X_list_np, axis=0)
        
        # Sparse k-NN graph on CPU
        k = min(self.k_neighbors, n_samples - 1)
        A = kneighbors_graph(X_np.T, n_neighbors=k, mode='connectivity', include_self=False)
        A = ((A + A.T) > 0).astype(float).toarray()
        
        same = (y[:, None] == y[None, :])
        W_w = A * same
        L_w_np = np.diag(W_w.sum(1)) - W_w
        
        # Move to GPU
        L_w = torch.tensor(L_w_np, dtype=torch.float32, device=device)
        X_list = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_list_np]
        
        P_blocks = []
        for i in range(m):
            pca = PCA(n_components=p, random_state=42)
            pca.fit(X_list_np[i].T)
            P_blocks.append(torch.tensor(pca.components_.T, dtype=torch.float32, device=device))

        J = torch.zeros((n_samples, n_samples), dtype=torch.float32, device=device)
        Z = torch.zeros((n_samples, n_samples), dtype=torch.float32, device=device)
        E = torch.zeros((mp, n_samples), dtype=torch.float32, device=device)

        Y1_blocks = [torch.zeros((p, n_samples), dtype=torch.float32, device=device) for _ in range(m)]
        Y3_blocks = [torch.zeros((d_list[i], p), dtype=torch.float32, device=device) for i in range(m)]
        Q_blocks = [P_blocks[i].clone() for i in range(m)]
        
        Y2 = torch.zeros((n_samples, n_samples), dtype=torch.float32, device=device)

        mu = 1e-6
        max_mu = 1e6
        rho = 1.2
        eye_n = torch.eye(n_samples, dtype=torch.float32, device=device)
        
        for iteration in range(self.max_iter):
            PtX_blocks = [P_blocks[i].T @ X_list[i] for i in range(m)]
            PtX = torch.cat(PtX_blocks, dim=0)
            
            J = svt_torch(Z + Y2 / mu, 1.0 / mu)
            Q_blocks = [svt_torch(P_blocks[i] + Y3_blocks[i] / mu, self.lambda1 / mu) for i in range(m)]
            
            Psi = PtX - PtX @ Z + torch.cat(Y1_blocks, dim=0) / mu
            E = shrinkage_torch(Psi, self.lambda0 / mu)
            
            lhs = PtX.T @ PtX + eye_n
            lhs = torch.nan_to_num(lhs, nan=0.0, posinf=1e6, neginf=-1e6) + 1e-8 * eye_n
            rhs = (PtX.T @ (PtX - E) + J + (PtX.T @ torch.cat(Y1_blocks, dim=0) + Y2) / mu)
            rhs = torch.nan_to_num(rhs, nan=0.0, posinf=1e6, neginf=-1e6)
            
            try:
                Z = torch.linalg.solve(lhs, rhs)
            except RuntimeError:
                Z = torch.linalg.lstsq(lhs, rhs).solution
                
            E_blocks = [E[i*p:(i+1)*p, :] for i in range(m)]
            PtXZ_blk = [PtX_blocks[i] @ Z for i in range(m)]
            
            for i in range(m):
                di = d_list[i]
                Xi = X_list[i]
                
                A_i = Xi @ L_w @ Xi.T
                cross_i = sum([X_list[i] @ X_list[j].T @ P_blocks[j] for j in range(m) if j != i])
                
                cov_i = Xi @ Xi.T
                rnorms_i = torch.linalg.norm(cov_i, dim=1)
                rnorms_i = torch.where(rnorms_i == 0, torch.tensor(1.0, device=device), rnorms_i)
                R_i = torch.diag(1.0 / (2.0 * rnorms_i))
                
                eye_di = torch.eye(di, dtype=torch.float32, device=device)
                M_i = (2 * self.lambda2 * A_i + mu * Xi @ Xi.T + mu * eye_di - 2 * self.lambda3 * self.delta * R_i)
                M_i = torch.nan_to_num(M_i, nan=0.0, posinf=1e6, neginf=-1e6) + 1e-8 * eye_di
                
                RHS_i = (Xi @ Y1_blocks[i].T + Y3_blocks[i] + 2 * self.lambda3 * cross_i - mu * Xi @ (E_blocks[i] - PtX_blocks[i] + PtXZ_blk[i]).T)
                RHS_i = torch.nan_to_num(RHS_i, nan=0.0, posinf=1e6, neginf=-1e6)
                
                try:
                    P_blocks[i] = torch.linalg.solve(M_i, RHS_i)
                except RuntimeError:
                    P_blocks[i] -= 0.01 * (M_i @ P_blocks[i] - RHS_i)
                    
            PtX_blocks = [P_blocks[i].T @ X_list[i] for i in range(m)]
            PtX = torch.cat(PtX_blocks, dim=0)
            
            diff1_blocks = [PtX_blocks[i] - PtX_blocks[i] @ Z - E_blocks[i] for i in range(m)]
            diff2 = Z - J
            diff3_blocks = [P_blocks[i] - Q_blocks[i] for i in range(m)]
            
            for i in range(m):
                Y1_blocks[i] += mu * diff1_blocks[i]
                Y3_blocks[i] += mu * diff3_blocks[i]
            Y2 += mu * diff2
            mu = min(rho * mu, max_mu)
            
            errs = [torch.linalg.norm(d) for d in diff1_blocks] + [torch.linalg.norm(diff2)] + [torch.linalg.norm(d) for d in diff3_blocks]
            if max(errs) < self.tol:
                break
                
        self.P_list_np = [P.cpu().numpy() for P in P_blocks]
        return self

    def transform(self, X_list_np):
        return np.concatenate([self.P_list_np[i].T @ X_list_np[i] for i in range(self.n_views)], axis=0)
