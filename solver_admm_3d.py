import numpy as np
import warnings

# ============================================================
# 0) Backend: try GPU (CuPy), fallback to CPU (NumPy)
# ============================================================
try:
    import cupy as cp
    _CUPY_OK = True
except Exception:
    cp = None
    _CUPY_OK = False


class Backend:
    """
    Backend wrapper:
      - backend.xp is numpy or cupy
      - to_device/to_cpu move arrays
      - scalar() safely converts xp scalar -> python float
    """
    def __init__(self, prefer_gpu=True, dtype=np.float32):
        self.prefer_gpu = prefer_gpu
        self.dtype = dtype

        if prefer_gpu and _CUPY_OK:
            self.xp = cp
            self.on_gpu = True
        else:
            self.xp = np
            self.on_gpu = False

    def to_device(self, a):
        if self.on_gpu:
            return cp.asarray(a, dtype=self.dtype)
        return np.asarray(a, dtype=self.dtype)

    def to_cpu(self, a):
        if self.on_gpu:
            return cp.asnumpy(a)
        return np.asarray(a)

    def scalar(self, x):
        if self.on_gpu:
            return float(cp.asnumpy(x))
        return float(x)

    def print_device_info(self):
        if not self.on_gpu:
            print("Using CPU backend (NumPy).")
            return
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode("utf-8") if isinstance(props["name"], (bytes, bytearray)) else str(props["name"])
        total_mem = props["totalGlobalMem"] / (1024**3)
        print(f"Using GPU backend (CuPy). Device: {name}, VRAM ~ {total_mem:.2f} GB")


# ============================================================
# 1) Discrete 3D operators (Z,Y,X)
# ============================================================
def grad_forward_3d(xp, u):
    """
    Forward differences with zero-Neumann boundary.
    u shape: (Z, Y, X)
    returns: (gz, gy, gx)
    """
    gz = xp.zeros_like(u)
    gy = xp.zeros_like(u)
    gx = xp.zeros_like(u)

    gz[:-1, :, :] = u[1:, :, :] - u[:-1, :, :]
    gy[:, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
    gx[:, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    return gz, gy, gx


def divergence_3d(xp, pz, py, px):
    """
    Adjoint of grad_forward_3d.
    """
    div = xp.zeros_like(px)

    # x
    div[:, :, 0] = px[:, :, 0]
    div[:, :, 1:-1] = px[:, :, 1:-1] - px[:, :, :-2]
    div[:, :, -1] = -px[:, :, -2]

    # y
    div[:, 0, :] += py[:, 0, :]
    div[:, 1:-1, :] += py[:, 1:-1, :] - py[:, :-2, :]
    div[:, -1, :] += -py[:, -2, :]

    # z
    div[0, :, :] += pz[0, :, :]
    div[1:-1, :, :] += pz[1:-1, :, :] - pz[:-2, :, :]
    div[-1, :, :] += -pz[-2, :, :]

    return div


# ============================================================
# 2) z-update (Newton) in 3D (voxelwise)
# ============================================================
def z_update_newton_3d(
    backend: Backend,
    q,
    b,
    mu: float,
    beta: float,
    z0=None,
    max_iters=30,
    tol=1e-6,
    admm_k=None,
):
    """
    Solve voxelwise for z>0:
        beta z^3 - beta q z^2 + mu z - mu b = 0
    """
    xp = backend.xp
    if z0 is None:
        z = xp.maximum(q, 1e-12)
    else:
        z = xp.maximum(z0, 1e-12)

    max_update_val = np.inf
    for _ in range(max_iters):
        g = beta * z**3 - beta * q * z**2 + mu * z - mu * b
        g_prime = 3 * beta * z**2 - 2 * beta * q * z + mu
        g_prime = xp.where(xp.abs(g_prime) < 1e-12, 1e-12, g_prime)

        step = g / g_prime
        step = xp.clip(step, -0.5 * z, 0.5 * z)  # damping
        z_new = xp.maximum(z - step, 1e-12)

        max_update_val = backend.scalar(xp.max(xp.abs(z_new - z)))
        z = z_new
        if max_update_val < tol:
            return z

    warnings.warn(
        f"Newton z-update (3D): no convergence (max_update={max_update_val:.3e} > tol={tol})"
        + (f" [ADMM k={admm_k}, Newton it={max_iters}]" if admm_k is not None else ""),
        RuntimeWarning,
    )
    return z


# ============================================================
# 3) u-update: 3D TV + quadratic via Chambolle–Pock
#     ||∇||^2 <= 12 in 3D => L = sqrt(12)
# ============================================================
def tv_denoise_primal_dual_3d(
    backend: Backend,
    v,
    alpha,          # scalar (recommended) or array
    beta,
    n_iters=80,
    tau=None,
    sigma=None,
    theta=1.0,
    tol=1e-4,
    verbose=False,
    check_every=5,
    admm_k=None,
):
    xp = backend.xp

    if tau is None or sigma is None:
        L = np.sqrt(12.0)
        tau = 0.99 / L
        sigma = 0.99 / L

    u = v.copy()
    u_bar = u.copy()

    pz = xp.zeros_like(v)
    py = xp.zeros_like(v)
    px = xp.zeros_like(v)

    if np.isscalar(alpha):
        alpha_is_scalar = True
        alpha_val = float(alpha)
        if alpha_val <= 0:
            alpha_val = 1e-12
    else:
        alpha_is_scalar = False
        alpha = xp.maximum(alpha, 1e-12)

    last_max_update = np.inf

    for it in range(n_iters):
        gz, gy, gx = grad_forward_3d(xp, u_bar)
        pz_new = pz + sigma * gz
        py_new = py + sigma * gy
        px_new = px + sigma * gx

        norm = xp.sqrt(pz_new**2 + py_new**2 + px_new**2)
        if alpha_is_scalar:
            denom = xp.maximum(1.0, norm / alpha_val)
        else:
            denom = xp.maximum(1.0, norm / alpha)

        pz = pz_new / denom
        py = py_new / denom
        px = px_new / denom

        div_p = divergence_3d(xp, pz, py, px)
        u_old = u
        u = (u + tau * div_p + tau * beta * v) / (1.0 + tau * beta)

        # positivity for log-term
        u = xp.maximum(u, 1e-6)

        u_bar = u + theta * (u - u_old)

        if (it % check_every == 0) or (it == n_iters - 1):
            last_max_update = backend.scalar(xp.max(xp.abs(u - u_old)))
            if verbose and (it % 20 == 0 or it < 5):
                print(f"  PD3D it {it+1:4d} | max_update={last_max_update:.3e} | ADMM k={admm_k}")
            if last_max_update < tol:
                return u

    warnings.warn(
        f"Primal-dual (3D): no convergence (max_update={last_max_update:.3e} > tol={tol})"
        + (f" [ADMM k={admm_k}, PD it={n_iters}]" if admm_k is not None else ""),
        RuntimeWarning,
    )
    return u


# ============================================================
# 4) ADMM (scaled) in 3D
# ============================================================
def admm_speckle_scaled_3d(
    backend: Backend,
    b,
    alpha,
    mu: float,
    beta: float = 700.0,
    n_admm_iters: int = 300,
    n_pd_iters: int = 80,
    pd_tol: float = 1e-4,
    admm_tol: float = 1e-4,
    auto_beta: bool = True,
    beta_mult: float = 2.0,
    balance_mu: float = 10.0,
    verbose: bool = False,
    state=None,
):
    xp = backend.xp

    if state is None:
        u = b.copy()
        z = b.copy()
        d = xp.zeros_like(b)
        beta_k = float(beta)
    else:
        u = state["u"]
        z = state["z"]
        d = state["d"]
        beta_k = float(state.get("beta", beta))

    n = int(b.size)
    hist = []

    for k in range(n_admm_iters):
        z_prev = z.copy()

        v = z - d
        u = tv_denoise_primal_dual_3d(
            backend,
            v=v,
            alpha=alpha,
            beta=beta_k,
            n_iters=n_pd_iters,
            tol=pd_tol,
            tau=0.99 / np.sqrt(12.0),
            sigma=0.99 / np.sqrt(12.0),
            verbose=False,
            check_every=5,
            admm_k=k,
        )

        q = u + d
        z = z_update_newton_3d(
            backend,
            q=q,
            b=b,
            mu=float(mu),
            beta=float(beta_k),
            z0=z_prev,
            max_iters=30,
            tol=1e-6,
            admm_k=k,
        )

        d = d + (u - z)

        r = u - z
        s = beta_k * (z - z_prev)

        r_norm = backend.scalar(xp.linalg.norm(r))
        s_norm = backend.scalar(xp.linalg.norm(s))

        u_norm = backend.scalar(xp.linalg.norm(u))
        z_norm = backend.scalar(xp.linalg.norm(z))
        bd_norm = backend.scalar(xp.linalg.norm(beta_k * d))

        eps_abs = admm_tol
        eps_rel = admm_tol
        eps_pri = np.sqrt(n) * eps_abs + eps_rel * max(u_norm, z_norm)
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * bd_norm

        if auto_beta:
            old_beta = beta_k
            if r_norm > balance_mu * s_norm:
                beta_k *= beta_mult
            elif s_norm > balance_mu * r_norm:
                beta_k /= beta_mult
            if beta_k != old_beta:
                d *= (old_beta / beta_k)

        if verbose and (k % 10 == 0 or k < 5):
            print(
                f"ADMM3D {k:4d} | beta={beta_k:.3g} | "
                f"||r||={r_norm:.3e} (<= {eps_pri:.3e}) | "
                f"||s||={s_norm:.3e} (<= {eps_dual:.3e})"
            )

        hist.append((beta_k, r_norm, s_norm, eps_pri, eps_dual))

        if (r_norm <= eps_pri) and (s_norm <= eps_dual) and k > 0:
            break

    out_state = {"u": u, "z": z, "d": d, "beta": beta_k}
    return u, out_state, np.array(hist, dtype=np.float64)


__all__ = [
    "Backend",
    "admm_speckle_scaled_3d",
    "tv_denoise_primal_dual_3d",
    "z_update_newton_3d",
]
