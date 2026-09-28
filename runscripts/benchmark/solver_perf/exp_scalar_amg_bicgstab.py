"""Offline test: AMG (frozen epot interpolation + Galerkin refresh) as right preconditioner
for BiCGStab on the captured scalar and velocity advection systems (python reference)."""
import math, sys, torch
from phipict.solvers import amg
D = "output/solver_perf/systems/HartmannSmall3D-downward-easy-v0/"
base = torch.load(D + "epot_amg_hierarchy.pt", weights_only=False)

def bicgstab(A, b, x, M, tol, maxit=500):
    n = b.numel()
    r = b - A @ x; r0 = r.clone(); rho = alpha = omega = 1.0
    v = torch.zeros_like(b); p = torch.zeros_like(b)
    for it in range(maxit):
        rho_new = torch.dot(r0, r).item()
        beta = (rho_new / rho) * (alpha / omega) if it else 0.0
        p = r + beta * (p - omega * v) if it else r.clone()
        ph = M(p); v = A @ ph
        alpha = rho_new / torch.dot(r0, v).item()
        s = r - alpha * v
        if torch.linalg.vector_norm(s).item() / math.sqrt(n) < tol:
            return x + alpha * ph, it, True
        sh = M(s); t = A @ sh
        omega = torch.dot(t, s).item() / torch.dot(t, t).item()
        x = x + alpha * ph + omega * sh
        r = s - omega * t; rho = rho_new
        if torch.linalg.vector_norm(r).item() / math.sqrt(n) < tol:
            return x, it + 1, True
    return x, maxit, False

for name in sys.argv[1:]:
    d = torch.load(D + name, weights_only=False)
    n = d["row"].numel() - 1
    A = torch.sparse_csr_tensor(d["row"].long().cuda(), d["index"].long().cuda(), d["value"].cuda(), (n, n))
    h = amg.galerkin_refresh(base, A, project_constant=False)
    nrhs = d["rhs"].numel() // n
    for k in range(nrhs):
        b = d["rhs"].cuda()[k*n:(k+1)*n]; x0 = d["x0"].cuda()[k*n:(k+1)*n]
        tol = d["meta"]["args"]["tol"]
        Av = lambda v: (A @ v.unsqueeze(1)).squeeze(1)
        class Op:
            def __matmul__(self, v): return Av(v)
        _, it_none, c0 = bicgstab(Op(), b, x0, lambda v: v, tol)
        dinv = amg._inverse_diagonal(A)
        _, it_jac, c1 = bicgstab(Op(), b, x0, lambda v: dinv * v, tol)
        _, it_amg, c2 = bicgstab(Op(), b, x0, h, tol)
        print(f"{name} rhs{k}: none {it_none} ({c0}), jacobi {it_jac} ({c1}), amg {it_amg} ({c2}); in-sim legacy {d['meta']['iters']}")
