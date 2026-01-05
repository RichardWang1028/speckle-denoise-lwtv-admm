import numpy as np
import warnings

"""
2D ADMM + primal-dual TV solver for LWTV-Log (speckle) model.
Supports NumPy (CPU) and CuPy (GPU).
"""

try:
    import cupy as cp
    _CUPY_OK = True
except Exception:
    cp = None
    _CUPY_OK = False


class Backend:
    """
    backend.xp is numpy or cupy.
    backend.to_device/to_cpu moves arrays.
    """
    def __init__(self, prefer_gpu=True, dtype=np.float32):
        self.prefer_gpu = prefer_gpu
        self.dtype = dtype

        if prefer_gpu and not _CUPY_OK:
            raise RuntimeError(
                "prefer_gpu=True but CuPy failed to import.\n"
                "Fix: install a CUDA-matching CuPy build (e.g. cupy-cuda12x) "
                "and ensure NVIDIA driver/CUDA runtime are OK.\n"
                "Or set prefer_gpu=False."
            )

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
# Discrete operators (2D)
# ============================================================
def grad_forward(xp, u):
    gx = xp.zeros_like(u)
    gy = xp.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def divergence(xp, px, py):
    div = xp.zeros_like(px)
    div[:, 0] = px[:, 0]
    div[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    div[:, -1] = -px[:, -2]

    div[0, :] += py[0, :]
    div[1:-1, :] += py[1:-1, :] - py[:-2, :]
    div[-1, :] += -py[-2, :]
    return div


# ============================================================
# z-update (Newton) — pixelwise cubic
# ============================================================
def z_update_newton(backend: Backend, q, b, mu, beta, z0=None, max_iters=30, tol=1e-6, admm_k=None):
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
        step = xp.clip(step, -0.5 * z, 0.5 * z)
        z_new = xp.maximum(z - step, 1e-12)

        max_update_val = backend.scalar(xp.max(xp.abs(z_new - z)))
        z = z_new
        if max_update_val < tol:
            return z

    warnings.warn(
        f"Newton z-update: no convergence (max_update={max_update_val:.3e} > tol={tol})"
        + (f" [ADMM k={admm_k}]" if admm_k is not None else ""),
        RuntimeWarning,
    )
    return z


# ============================================================
# u-update: weighted TV + quadratic via Chambolle–Pock
#   min_u  sum_i alpha_i ||(∇u)_i||_2 + (beta/2)||u - v||^2
# ============================================================
def tv_denoise_primal_dual(
    backend: Backend,
    v,
    alpha,         # scalar or array (H,W)
    beta,
    n_iters=80,
    tau=None,
    sigma=None,
    theta=1.0,
    tol=1e-4,
    check_every=5,
    verbose=False,
    admm_k=None,
):
    xp = backend.xp

    # ||∇||^2 <= 8 in 2D forward differences
    if tau is None or sigma is None:
        L = np.sqrt(8.0)
        tau = 0.99 / L
        sigma = 0.99 / L

    u = v.copy()
    u_bar = u.copy()
    px = xp.zeros_like(v)
    py = xp.zeros_like(v)

    if np.isscalar(alpha):
        alpha_is_scalar = True
        alpha_val = float(alpha)
        if alpha_val <= 0:
            alpha_val = 1e-12
    else:
        alpha_is_scalar = False
        # CRITICAL for GPU: ensure alpha is on the same device
        alpha = backend.to_device(alpha)
        alpha = xp.maximum(alpha, 1e-12)

    last_max_update = np.inf

    for it in range(n_iters):
        gx, gy = grad_forward(xp, u_bar)
        px_new = px + sigma * gx
        py_new = py + sigma * gy

        norm = xp.sqrt(px_new**2 + py_new**2)
        if alpha_is_scalar:
            denom = xp.maximum(1.0, norm / alpha_val)
        else:
            denom = xp.maximum(1.0, norm / alpha)

        px = px_new / denom
        py = py_new / denom

        div_p = divergence(xp, px, py)
        u_old = u
        u = (u + tau * div_p + tau * beta * v) / (1.0 + tau * beta)

        # positivity for log term
        u = xp.maximum(u, 1e-6)

        u_bar = u + theta * (u - u_old)

        if (it % check_every == 0) or (it == n_iters - 1):
            last_max_update = backend.scalar(xp.max(xp.abs(u - u_old)))
            if verbose and (it % 20 == 0 or it < 5):
                print(f"  PD it {it+1:4d} | max_update={last_max_update:.3e} | ADMM k={admm_k}")
            if last_max_update < tol:
                return u

    warnings.warn(
        f"Primal-dual: no convergence (max_update={last_max_update:.3e} > tol={tol})"
        + (f" [ADMM k={admm_k}, PD it={n_iters}]" if admm_k is not None else ""),
        RuntimeWarning,
    )
    return u


# ============================================================
# ADMM (scaled dual)
# ============================================================
def admm_speckle_scaled(
    backend: Backend,
    b,
    alpha,          # scalar OR alpha-map
    mu,
    beta=700.0,
    n_admm_iters=300,
    n_pd_iters=80,
    pd_tol=1e-4,
    admm_tol=1e-4,
    auto_beta=True,
    beta_mult=2.0,
    balance_mu=10.0,
    verbose=False,
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

        # u-update
        v = z - d
        u = tv_denoise_primal_dual(
            backend,
            v=v,
            alpha=alpha,          # <-- scalar or map
            beta=beta_k,
            n_iters=n_pd_iters,
            tol=pd_tol,
            tau=0.99 / np.sqrt(8.0),
            sigma=0.99 / np.sqrt(8.0),
            check_every=5,
            verbose=False,
            admm_k=k,
        )

        # z-update
        q = u + d
        z = z_update_newton(
            backend, q, b, mu, beta_k, z0=z_prev,
            max_iters=30, tol=1e-6, admm_k=k
        )

        # dual update
        d = d + (u - z)

        # residuals
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
                f"ADMM {k:4d} | beta={beta_k:.3g} | "
                f"||r||={r_norm:.3e} (<= {eps_pri:.3e}) | "
                f"||s||={s_norm:.3e} (<= {eps_dual:.3e})"
            )

        hist.append((beta_k, r_norm, s_norm, eps_pri, eps_dual))

        if (r_norm <= eps_pri) and (s_norm <= eps_dual) and k > 0:
            break

    out_state = {"u": u, "z": z, "d": d, "beta": beta_k}
    return u, out_state, np.array(hist, dtype=np.float64)


__all__ = ["Backend", "admm_speckle_scaled"]
