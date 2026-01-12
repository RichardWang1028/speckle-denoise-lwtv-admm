# denoise_admm_with_alpha.py
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Project modules (must exist in the same project)
from alpha_net import (
    load_gray01,
    add_speckle_lognormal,
    AlphaUNetSmall,
    alpha_from_logits,
    make_alpha_target_from_clean,
    gradmag_np,
)

from solver_admm_2d import Backend, admm_speckle_scaled


# ============================================================
# I/O helpers
# ============================================================
def save_gray01_png(u01: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u = np.asarray(u01, dtype=np.float32)
    u = np.clip(u, 0.0, 1.0)
    im = (255.0 * u + 0.5).astype(np.uint8)
    Image.fromarray(im).save(str(out_path))


def _csv_floats(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


@contextmanager
def tee_console_to_file(log_path: Path):
    """
    Mirror stdout/stderr into a log file (reproducibility).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, "w", encoding="utf-8") as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# ============================================================
# Metrics (ONLY SSIM + ISNR)
# ============================================================
def isnr(u_true: np.ndarray, b: np.ndarray, u_hat: np.ndarray, eps: float = 1e-12) -> float:
    """
    ISNR [dB] = 20 log10( ||u_true - b|| / ||u_true - u_hat|| )
    Larger is better.
    """
    u_true = np.asarray(u_true, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    num = np.linalg.norm(u_true - b)
    den = np.linalg.norm(u_true - u_hat)
    den = max(den, eps)
    num = max(num, eps)
    return float(20.0 * np.log10(num / den))


def ssim_metric(u_true: np.ndarray, u_hat: np.ndarray) -> float:
    """
    SSIM in [0,1], larger is better.
    Requires scikit-image.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception as e:
        raise RuntimeError("SSIM requires scikit-image. Install with: pip install scikit-image") from e

    u_true = np.asarray(u_true, dtype=np.float32)
    u_hat = np.asarray(u_hat, dtype=np.float32)
    return float(ssim(u_true, u_hat, data_range=1.0))


# ============================================================
# Diagnostics
# ============================================================
def alpha_stats(alpha: np.ndarray) -> Dict[str, float]:
    a = np.asarray(alpha, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def corr_alpha_grad(alpha: np.ndarray, img: np.ndarray) -> float:
    """
    Correlation corr(alpha, |∇img|).
    """
    a = np.asarray(alpha, dtype=np.float64).ravel()
    g = gradmag_np(np.asarray(img, dtype=np.float64)).ravel()
    a = a - a.mean()
    g = g - g.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(g)) + 1e-12
    return float(np.dot(a, g) / denom)


# ============================================================
# Plotting
# ============================================================
def plot_compare_with_gt(
    u_true: np.ndarray,
    b_vis: np.ndarray,
    u_hat: np.ndarray,
    alpha_vis: np.ndarray,
    out_path: Path,
    title_mid: str,
) -> None:
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1); plt.imshow(u_true, cmap="gray", vmin=0, vmax=1); plt.title("Clean"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(b_vis,  cmap="gray", vmin=0, vmax=1); plt.title("Noisy"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(u_hat,  cmap="gray", vmin=0, vmax=1); plt.title(title_mid); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(alpha_vis, cmap="gray"); plt.title("alpha-map"); plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_compare_no_gt(
    b_vis: np.ndarray,
    u_hat: np.ndarray,
    alpha_vis: np.ndarray,
    out_path: Path,
    title_mid: str,
) -> None:
    fig = plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(b_vis, cmap="gray", vmin=0, vmax=1); plt.title("Input/Noisy"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(u_hat, cmap="gray", vmin=0, vmax=1); plt.title(title_mid); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(alpha_vis, cmap="gray"); plt.title("alpha-map"); plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_compare_all(
    out_path: Path,
    u_true: Optional[np.ndarray],
    b_vis: np.ndarray,
    denoised_items: Sequence[Tuple[str, np.ndarray]],
) -> None:
    """
    Create a single 'compare_all.png' containing:
    Clean (if available), Noisy, then each denoised.
    """
    imgs: List[np.ndarray] = []
    titles: List[str] = []

    if u_true is not None:
        imgs.append(np.clip(u_true, 0, 1))
        titles.append("Clean")
    imgs.append(np.clip(b_vis, 0, 1))
    titles.append("Noisy")

    for name, u in denoised_items:
        imgs.append(np.clip(u, 0, 1))
        titles.append(name)

    n = len(imgs)
    fig = plt.figure(figsize=(3 * n, 3))
    for i, (im, t) in enumerate(zip(imgs, titles), 1):
        plt.subplot(1, n, i)
        plt.imshow(im, cmap="gray", vmin=0, vmax=1)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Config + run directory
# ============================================================
@dataclass
class RunConfig:
    img: str
    ckpt: str
    out_root: str
    tag: str

    # ADMM penalty mode (DEFAULT: fixed beta; enable adaptive by --auto_beta)
    auto_beta: bool
    beta_mult: float
    balance_mu: float

    synthetic: bool
    var: float
    seed: int
    cache_noise: bool

    # Single run params
    mu: float
    beta0: float

    # Sweep params (optional)
    mu_list: Optional[List[float]]
    beta_list: Optional[List[float]]

    # Solver
    admm: int
    pd: int
    prefer_gpu: bool

    # Baselines
    alpha_min: float
    alpha_max: float
    k: float
    alpha_const: Optional[float]
    run_const: bool
    run_oracle: bool


def make_run_dir(out_root: Path, tag: str, img_path: Path, cfg: RunConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = img_path.stem
    syn = f"syn_var{cfg.var:g}_seed{cfg.seed}" if cfg.synthetic else "real"
    pen = "autobeta" if cfg.auto_beta else "fixedbeta"

    if cfg.mu_list is not None or cfg.beta_list is not None:
        mN = len(cfg.mu_list or [cfg.mu])
        bN = len(cfg.beta_list or [cfg.beta0])
        hp = f"SWEEP_mu{mN}_beta{bN}_{pen}"
    else:
        hp = f"mu{cfg.mu:g}_beta{cfg.beta0:g}_{pen}"

    name = f"{tag}_{hp}_{stem}_{syn}_{ts}"
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================
# Core runner
# ============================================================
def run_one_method(
    method_label: str,
    out_dir: Path,
    backend: Backend,
    b: np.ndarray,
    b_dev: Any,
    alpha: np.ndarray | float,
    u_true: Optional[np.ndarray],
    mu: float,
    beta0: float,
    n_admm_iters: int,
    n_pd_iters: int,
    auto_beta: bool,
    beta_mult: float,
    balance_mu: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # alpha visualization
    if isinstance(alpha, (float, int)):
        a_vis = np.full_like(b, float(alpha), dtype=np.float32)
        alpha_for_solver = float(alpha)
    else:
        a_vis = np.asarray(alpha, dtype=np.float32)
        alpha_for_solver = backend.to_device(a_vis)

    # Run ADMM
    t0 = time.time()
    u_dev, state, hist = admm_speckle_scaled(
        backend,
        b=b_dev,
        alpha=alpha_for_solver,
        mu=float(mu),
        beta=float(beta0),
        n_admm_iters=int(n_admm_iters),
        n_pd_iters=int(n_pd_iters),
        auto_beta=bool(auto_beta),
        beta_mult=float(beta_mult),
        balance_mu=float(balance_mu),
        verbose=False,
    )
    t_sec = time.time() - t0

    u_hat = backend.to_cpu(u_dev).astype(np.float32)
    u_hat = np.clip(u_hat, 0.0, 1.0)
    b_vis = np.clip(b, 0.0, 1.0)

    save_gray01_png(b_vis, out_dir / "noisy.png")
    save_gray01_png(u_hat, out_dir / "denoised.png")
    a_norm = (a_vis - a_vis.min()) / (a_vis.max() - a_vis.min() + 1e-12)
    save_gray01_png(a_norm, out_dir / "alpha_vis.png")

    # ADMM history
    hist = np.asarray(hist)
    if hist.ndim == 2 and hist.shape[1] >= 5:
        hist_with_k = np.column_stack([np.arange(hist.shape[0], dtype=np.int32), hist[:, :5]])
        np.savetxt(
            out_dir / "admm_hist.csv",
            hist_with_k,
            delimiter=",",
            header="k,beta,r_norm,s_norm,eps_pri,eps_dual",
            comments="",
        )
        last = hist[-1, :5]
        r_norm, s_norm, eps_pri, eps_dual = float(last[1]), float(last[2]), float(last[3]), float(last[4])
        converged = (r_norm <= eps_pri) and (s_norm <= eps_dual)
        admm_iters_used = int(hist.shape[0])
        beta_final = float(last[0])
    else:
        converged = False
        admm_iters_used = int(n_admm_iters)
        beta_final = float(beta0)

    # Compare figure + metrics
    metrics: Dict[str, Any] = {
        "method": method_label,
        "mu": float(mu),
        "beta0": float(beta0),
        "auto_beta": bool(auto_beta),
        "beta_mult": float(beta_mult),
        "balance_mu": float(balance_mu),
        "beta_final": float(beta_final),
        "runtime_sec": float(t_sec),
        "admm_iters_target": int(n_admm_iters),
        "admm_iters_used": int(admm_iters_used),
        "pd_iters": int(n_pd_iters),
        "converged": bool(converged),
        "out_dir": str(out_dir.name) if out_dir.parent is None else str(out_dir.relative_to(out_dir.parents[0])),
    }

    if u_true is not None:
        u_true_vis = np.clip(u_true, 0.0, 1.0)
        plot_compare_with_gt(u_true_vis, b_vis, u_hat, a_vis, out_dir / "compare.png", title_mid=method_label)
        metrics["ssim"] = ssim_metric(u_true_vis, u_hat)
        metrics["isnr"] = isnr(u_true_vis, b_vis, u_hat)
    else:
        plot_compare_no_gt(b_vis, u_hat, a_vis, out_dir / "compare.png", title_mid=method_label)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _load_alpha_ckpt_into_net(net: torch.nn.Module, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Robust checkpoint loader.

    Supports:
      - {"state_dict": <weights>, "alpha_min":..., "alpha_max":..., "k":...}
      - {"model_state_dict": <weights>, ...}
      - {"model": <weights>, ...}
      - <weights> (plain state_dict)
    """
    raw = torch.load(str(ckpt_path), map_location=device)

    if not isinstance(raw, dict):
        raise RuntimeError("Unexpected checkpoint format: expected a dict checkpoint.")

    if "state_dict" in raw:
        sd = raw["state_dict"]
    elif "model_state_dict" in raw:
        sd = raw["model_state_dict"]
    elif "model" in raw:
        sd = raw["model"]
    else:
        # maybe it IS the state_dict
        sd = raw

    if not isinstance(sd, dict):
        raise RuntimeError("Unexpected checkpoint state_dict format (not a dict).")

    # Strip DataParallel prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    try:
        net.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        # Make the error more actionable.
        # (Do not silently set strict=False here; that can hide real mismatches.)
        raise RuntimeError(
            "Failed to load checkpoint weights into AlphaUNetSmall. "
            "This usually means the checkpoint was trained with a different architecture, "
            "or the wrong key was loaded (expected a state_dict). "
            f"Checkpoint: {ckpt_path}"
        ) from e

    return raw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: ADMM speckle denoising with learned alpha-map (SSIM + ISNR only). "
                    "Default is FIXED beta (thesis-consistent). Enable adaptive beta by --auto_beta."
    )
    ap.add_argument("--img", required=True, help="path to input image")
    ap.add_argument("--ckpt", required=True, help="path to alpha_net checkpoint (.pt)")
    ap.add_argument("--out_root", default="runs", help="root folder for run directories")
    ap.add_argument("--tag", default="step2", help="prefix tag for run folder name")

    ap.add_argument("--synthetic", action="store_true", help="generate synthetic speckle and evaluate with GT")
    ap.add_argument("--var", type=float, default=0.01, help="speckle variance (synthetic only)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed (synthetic only)")
    ap.add_argument("--cache_noise", action="store_true", help="cache synthetic noisy observation to disk")

    # single-run hyperparams
    ap.add_argument("--mu", type=float, default=2.0, help="data-fidelity weight mu")
    ap.add_argument("--beta0", type=float, default=700.0, help="ADMM penalty beta0 (initial/fixed)")

    # sweep hyperparams (optional)
    ap.add_argument("--mu_list", default=None, help="comma-separated sweep values, overrides --mu")
    ap.add_argument("--beta_list", default=None, help="comma-separated sweep values, overrides --beta0")

    ap.add_argument("--admm", type=int, default=200, help="ADMM iterations")
    ap.add_argument("--pd", type=int, default=80, help="primal-dual iterations per ADMM step")
    ap.add_argument("--prefer_gpu", action="store_true", help="use GPU backend via CuPy for ADMM (if available)")

    # IMPORTANT: default fixed beta; enable adaptive by flag
    ap.add_argument("--auto_beta", action="store_true", help="enable adaptive beta update (scaled ADMM heuristic)")
    ap.add_argument("--beta_mult", type=float, default=2.0, help="adaptive beta multiplier (when balancing residuals)")
    ap.add_argument("--balance_mu", type=float, default=10.0, help="residual-balance threshold (mu in literature)")

    ap.add_argument("--alpha_min", type=float, default=0.2, help="alpha lower bound")
    ap.add_argument("--alpha_max", type=float, default=2.0, help="alpha upper bound")
    ap.add_argument("--k", type=float, default=25.0, help="alpha shaping parameter (for oracle alpha target)")

    ap.add_argument("--alpha_const", type=float, default=None, help="constant alpha baseline (default=mean learned alpha)")
    ap.add_argument("--no_const", action="store_true", help="disable constant-alpha baseline")
    ap.add_argument("--no_oracle", action="store_true", help="disable oracle-alpha baseline (synthetic only)")

    args = ap.parse_args()

    cfg = RunConfig(
        img=str(args.img),
        ckpt=str(args.ckpt),
        out_root=str(args.out_root),
        tag=str(args.tag),

        auto_beta=bool(args.auto_beta),          # DEFAULT False (fixed beta)
        beta_mult=float(args.beta_mult),
        balance_mu=float(args.balance_mu),

        synthetic=bool(args.synthetic),
        var=float(args.var),
        seed=int(args.seed),
        cache_noise=bool(args.cache_noise),

        mu=float(args.mu),
        beta0=float(args.beta0),

        mu_list=_csv_floats(args.mu_list),
        beta_list=_csv_floats(args.beta_list),

        admm=int(args.admm),
        pd=int(args.pd),
        prefer_gpu=bool(args.prefer_gpu),

        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        k=float(args.k),

        alpha_const=None if args.alpha_const is None else float(args.alpha_const),
        run_const=not bool(args.no_const),
        run_oracle=not bool(args.no_oracle),
    )

    img_path = Path(cfg.img)
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(out_root, cfg.tag, img_path, cfg)
    log_path = run_dir / "run.log"

    with tee_console_to_file(log_path):
        print("=== Run directory ===")
        print(run_dir.resolve())
        print("=== Config ===")
        print(json.dumps(asdict(cfg), indent=2))
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        if cfg.auto_beta:
            print("=== ADMM penalty mode ===")
            print("Adaptive beta ENABLED (--auto_beta). This is a heuristic and not covered by fixed-penalty convergence theory.")
        else:
            print("=== ADMM penalty mode ===")
            print("Fixed beta (DEFAULT). This matches the convergence analysis assumptions in the thesis.")

        # Load input
        u_in = load_gray01(str(img_path))
        u_in = np.clip(np.asarray(u_in, dtype=np.float32), 1e-6, 1.0)

        u_true: Optional[np.ndarray] = None
        if cfg.synthetic:
            u_true = u_in

            if cfg.cache_noise:
                cache_dir = Path(cfg.out_root) / "_noise_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                H, W = u_true.shape
                cache_file = cache_dir / f"{img_path.stem}_{H}x{W}_var{cfg.var:g}_seed{cfg.seed}.npy"
                if cache_file.exists():
                    b = np.load(cache_file).astype(np.float32)
                    print("Synthetic noisy observation cache:", cache_file)
                else:
                    b = add_speckle_lognormal(u_true, var=cfg.var, seed=cfg.seed).astype(np.float32)
                    np.save(cache_file, b)
                    print("Synthetic noisy observation cached at:", cache_file)
            else:
                b = add_speckle_lognormal(u_true, var=cfg.var, seed=cfg.seed).astype(np.float32)

            save_gray01_png(u_true, run_dir / "clean.png")
            save_gray01_png(np.clip(b, 0, 1), run_dir / "noisy.png")
        else:
            b = u_in
            save_gray01_png(np.clip(b, 0, 1), run_dir / "input.png")

        b = np.clip(np.asarray(b, dtype=np.float32), 1e-6, 1.0)

        # Torch model (alpha-net)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch device:", device)
        if device.type == "cuda":
            print("Torch CUDA GPU:", torch.cuda.get_device_name(0))

        ckpt_path = Path(cfg.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        net = AlphaUNetSmall().to(device)
        net.eval()

        ckpt_meta = _load_alpha_ckpt_into_net(net, ckpt_path, device)

        # Infer learned alpha-map
        alpha_min, alpha_max, k = cfg.alpha_min, cfg.alpha_max, cfg.k
        x = torch.from_numpy(b[None, None, ...]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            logits = net(x)
            alpha_t = alpha_from_logits(logits, alpha_min, alpha_max)[0, 0]
        alpha_np = alpha_t.detach().cpu().numpy().astype(np.float32)

        print("=== Learned alpha diagnostics ===")
        st = alpha_stats(alpha_np)
        for kk, vv in st.items():
            print(f"{kk:>12s}: {vv}")
        try:
            c = corr_alpha_grad(alpha_np, b)
            print(f"{'corr(alpha,|∇b|)':>18s}: {c:+.4f}")
        except Exception as e:
            print("corr(alpha,|∇b|) failed:", repr(e))

        # Oracle alpha (only meaningful for synthetic, uses clean u_true)
        oracle_alpha_np: Optional[np.ndarray] = None
        if cfg.synthetic and cfg.run_oracle and (u_true is not None):
            oracle_alpha_np = make_alpha_target_from_clean(
                u_true, alpha_min=alpha_min, alpha_max=alpha_max, k=k
            ).astype(np.float32)
            print("=== Oracle alpha diagnostics (from clean u_true) ===")
            st2 = alpha_stats(oracle_alpha_np)
            for kk, vv in st2.items():
                print(f"{kk:>12s}: {vv}")

        # Constant alpha baseline
        alpha_const = cfg.alpha_const
        if alpha_const is None:
            alpha_const = float(alpha_np.mean())
            print(f"alpha_const not provided -> using mean(learned alpha) = {alpha_const:.6f}")
        else:
            print(f"alpha_const provided = {alpha_const:.6f}")

        # Backend + move noisy image once
        backend = Backend(prefer_gpu=cfg.prefer_gpu, dtype=np.float32)
        backend.print_device_info()
        b_dev = backend.to_device(b)

        # Build grids
        mu_grid = cfg.mu_list if cfg.mu_list is not None else [cfg.mu]
        beta_grid = cfg.beta_list if cfg.beta_list is not None else [cfg.beta0]
        do_sweep = (cfg.mu_list is not None) or (cfg.beta_list is not None)

        summary_rows: List[Dict[str, Any]] = []
        best_learned: Optional[Dict[str, Any]] = None
        best_pair: Tuple[float, float] = (mu_grid[0], beta_grid[0])

        # 1) learned alpha (single or sweep)
        print("=== Running learned_alpha ===")
        for mu in mu_grid:
            for beta0 in beta_grid:
                tag = f"learned_alpha_mu{mu:g}_beta{beta0:g}" if do_sweep else "learned_alpha"
                out_dir = run_dir / tag
                m = run_one_method(
                    method_label="learned_alpha",
                    out_dir=out_dir,
                    backend=backend,
                    b=b,
                    b_dev=b_dev,
                    alpha=alpha_np,
                    u_true=u_true,
                    mu=float(mu),
                    beta0=float(beta0),
                    n_admm_iters=cfg.admm,
                    n_pd_iters=cfg.pd,
                    auto_beta=cfg.auto_beta,
                    beta_mult=cfg.beta_mult,
                    balance_mu=cfg.balance_mu,
                )
                summary_rows.append(m)

                # best selection (only if GT exists -> can use SSIM)
                if u_true is not None and "ssim" in m:
                    if (best_learned is None) or (m["ssim"] > best_learned["ssim"]):
                        best_learned = m
                        best_pair = (float(mu), float(beta0))

        if do_sweep and u_true is not None and best_learned is not None:
            print(
                f"Best learned alpha combo (by SSIM): mu={best_pair[0]}, beta0={best_pair[1]}, "
                f"SSIM={best_learned['ssim']:.6f}, ISNR={best_learned['isnr']:.3f} dB"
            )
        elif do_sweep and u_true is None:
            print("Sweep requested but no GT (real image). I will NOT claim a 'best' (no SSIM/ISNR).")

        # 2) const alpha + oracle alpha: run once (single run) or at best_pair (sweep)
        mu0, beta00 = best_pair if do_sweep else (cfg.mu, cfg.beta0)

        if cfg.run_const:
            print("=== Running const_alpha ===")
            m = run_one_method(
                method_label="const_alpha",
                out_dir=run_dir / "const_alpha",
                backend=backend,
                b=b,
                b_dev=b_dev,
                alpha=float(alpha_const),
                u_true=u_true,
                mu=float(mu0),
                beta0=float(beta00),
                n_admm_iters=cfg.admm,
                n_pd_iters=cfg.pd,
                auto_beta=cfg.auto_beta,
                beta_mult=cfg.beta_mult,
                balance_mu=cfg.balance_mu,
            )
            summary_rows.append(m)

        if oracle_alpha_np is not None:
            print("=== Running oracle_alpha ===")
            m = run_one_method(
                method_label="oracle_alpha",
                out_dir=run_dir / "oracle_alpha",
                backend=backend,
                b=b,
                b_dev=b_dev,
                alpha=oracle_alpha_np,
                u_true=u_true,
                mu=float(mu0),
                beta0=float(beta00),
                n_admm_iters=cfg.admm,
                n_pd_iters=cfg.pd,
                auto_beta=cfg.auto_beta,
                beta_mult=cfg.beta_mult,
                balance_mu=cfg.balance_mu,
            )
            summary_rows.append(m)

        # Write summary.csv
        all_keys = sorted({k for row in summary_rows for k in row.keys()})
        with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)

        # Print concise summary (ONLY SSIM + ISNR)
        print("=== Summary (SSIM + ISNR only) ===")
        for row in summary_rows:
            meth = row.get("method", "")
            parts = [f"mu={row.get('mu')}", f"beta0={row.get('beta0')}"]
            parts.append(f"auto_beta={row.get('auto_beta')}")
            if "ssim" in row:
                parts.append(f"ssim={row['ssim']:.4f}")
            if "isnr" in row:
                parts.append(f"isnr={row['isnr']:.3f} dB")
            parts.append(f"iters={row.get('admm_iters_used')}/{row.get('admm_iters_target')}")
            parts.append(f"conv={row.get('converged')}")
            print(f"[{meth}] " + ", ".join(parts))

        # Compare-all image: use best learned (if sweep+GT), else learned_alpha single run
        denoised_items: List[Tuple[str, np.ndarray]] = []
        if u_true is not None:
            if do_sweep and best_learned is not None:
                best_tag = f"learned_alpha_mu{best_pair[0]:g}_beta{best_pair[1]:g}"
                den_path = run_dir / best_tag / "denoised.png"
                u_best = load_gray01(str(den_path))
                denoised_items.append((f"learned (best)", u_best))
            else:
                den_path = run_dir / "learned_alpha" / "denoised.png"
                if den_path.exists():
                    u_best = load_gray01(str(den_path))
                    denoised_items.append(("learned", u_best))

            if (run_dir / "const_alpha" / "denoised.png").exists():
                denoised_items.append(("const", load_gray01(str(run_dir / "const_alpha" / "denoised.png"))))
            if (run_dir / "oracle_alpha" / "denoised.png").exists():
                denoised_items.append(("oracle", load_gray01(str(run_dir / "oracle_alpha" / "denoised.png"))))

            make_compare_all(run_dir / "compare_all.png", u_true=u_true, b_vis=b, denoised_items=denoised_items)

        print("Done. Outputs saved to:", run_dir.resolve())


if __name__ == "__main__":
    main()
