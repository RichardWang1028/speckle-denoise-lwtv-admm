# compare_cine_vs_full3d.py
# Compare cine (2D-per-slice) vs full3d (3D volume) denoising on the SAME slice indices.
#
# Expected files:
#   cine_run/
#     denoised_cine.npy        shape (K,H,W)
#     frame_indices.npy        shape (K,) indices into original depth axis
#   full3d_run/
#     denoised_volume.npy      shape (T,H,W)  (T is full depth)
#
# Output:
#   out_dir/
#     compare_0000.png, ...
#     grid.png
#     roi_metrics.csv (optional if ROI args provided)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pydicom


# ----------------------------
# DICOM loading
# ----------------------------
def _infer_max_value(ds, arr: np.ndarray) -> float:
    bits = getattr(ds, "BitsStored", None)
    if bits is not None:
        try:
            bits = int(bits)
            if 1 <= bits <= 16:
                return float((1 << bits) - 1)
        except Exception:
            pass
    m = float(np.max(arr))
    return m if m > 0 else 1.0


def load_us_multiframe_as_float01(dicom_path: str | Path, eps: float = 1e-6):
    """
    Returns (ds, b01, scale) where b01 is float32 in [eps,1] with shape (T,H,W).
    """
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got {arr.shape}")
    scale = _infer_max_value(ds, arr)
    b01 = arr.astype(np.float32) / float(scale)
    b01 = np.clip(b01, eps, 1.0)
    return ds, b01, float(scale)


# ----------------------------
# ROI parsing + metrics
# ----------------------------
def parse_roi(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """
    ROI format: "y0,y1,x0,x1"
    """
    if s is None:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'y0,y1,x0,x1'")
    y0, y1, x0, x1 = parts
    if not (y0 < y1 and x0 < x1):
        raise ValueError("ROI must satisfy y0<y1 and x0<x1")
    return y0, y1, x0, x1


def _safe_crop(img01: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = roi
    H, W = img01.shape
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    if not (y0 < y1 and x0 < x1):
        return np.empty((0,), dtype=np.float64)
    return img01[y0:y1, x0:x1].astype(np.float64).ravel()


def enl(img01: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """
    ENL = mu^2 / sigma^2 on ROI, with sample sigma (ddof=1) consistent with your thesis.
    """
    v = _safe_crop(img01, roi)
    if v.size < 2:
        return float("nan")
    mu = float(np.mean(v))
    sigma = float(np.std(v, ddof=1))
    if sigma <= 0:
        return float("nan")
    return float((mu * mu) / (sigma * sigma))


def gcnr(img01: np.ndarray,
         roi_in: Tuple[int, int, int, int],
         roi_out: Tuple[int, int, int, int],
         bins: int = 256) -> float:
    """
    gCNR = 1 - OVL, OVL = ∫ min(p_in(t), p_out(t)) dt over [0,1],
    approximated by histogram overlap.
    """
    a = _safe_crop(img01, roi_in)
    b = _safe_crop(img01, roi_out)
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    pdf_a, edges = np.histogram(a, bins=bins, range=(0.0, 1.0), density=True)
    pdf_b, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=True)
    dt = float(edges[1] - edges[0])
    ovl = float(np.sum(np.minimum(pdf_a, pdf_b)) * dt)
    return float(np.clip(1.0 - ovl, 0.0, 1.0))


# ----------------------------
# Plotting helpers
# ----------------------------
def save_panel(out_png: Path,
               b2d: np.ndarray,
               cine2d: np.ndarray,
               full2d: np.ndarray,
               t: int,
               show_diff: bool = True):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    diff = np.abs(full2d - cine2d)

    ncols = 4 if show_diff else 3
    fig = plt.figure(figsize=(4.5 * ncols, 4.2))

    def _imshow(ax, img, title, vmin=0, vmax=1):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    ax1 = plt.subplot(1, ncols, 1)
    _imshow(ax1, b2d, f"Input b (frame {t})")

    ax2 = plt.subplot(1, ncols, 2)
    _imshow(ax2, cine2d, "Cine (2D)")

    ax3 = plt.subplot(1, ncols, 3)
    _imshow(ax3, full2d, "Full3D (3D)")

    if show_diff:
        ax4 = plt.subplot(1, ncols, 4)
        ax4.imshow(diff, cmap="gray")
        ax4.set_title("|Full3D − Cine|")
        ax4.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_grid(out_png: Path,
              frames: List[int],
              b01: np.ndarray,
              cine01: np.ndarray,
              full01: np.ndarray,
              show_diff: bool = True):
    """
    Make one big grid: rows = frames, columns = input/cine/full/(diff).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    K = len(frames)
    ncols = 4 if show_diff else 3
    fig = plt.figure(figsize=(4.2 * ncols, 2.8 * K))

    for i, t in enumerate(frames):
        b2d = b01[t]
        cine2d = cine01[i]
        full2d = full01[t]
        diff = np.abs(full2d - cine2d)

        r = i + 1
        ax = plt.subplot(K, ncols, (r - 1) * ncols + 1)
        ax.imshow(b2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"b (t={t})")
        ax.axis("off")

        ax = plt.subplot(K, ncols, (r - 1) * ncols + 2)
        ax.imshow(cine2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title("cine")
        ax.axis("off")

        ax = plt.subplot(K, ncols, (r - 1) * ncols + 3)
        ax.imshow(full2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title("full3d")
        ax.axis("off")

        if show_diff:
            ax = plt.subplot(K, ncols, (r - 1) * ncols + 4)
            ax.imshow(diff, cmap="gray")
            ax.set_title("|Δ|")
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Compare cine (2D) vs full3d (3D) on same frame indices.")
    ap.add_argument("--dicom", required=True, help="Original multi-frame DICOM path (used to get raw input slices)")
    ap.add_argument("--cine_run", required=True, help="Run folder from denoise_dicom_cine_CNN.py")
    ap.add_argument("--full3d_run", required=True, help="Run folder from denoise_dicom_full3d_CNN.py")
    ap.add_argument("--out_dir", default=None, help="Output folder (default: cine_run/compare_with_full3d)")
    ap.add_argument("--show_diff", action="store_true", help="Include |Full3D-Cine| panels")
    ap.add_argument("--frames_source", choices=["cine", "manual"], default="cine",
                    help="Use cine frame_indices.npy or manual list.")
    ap.add_argument("--frames", default=None, help="Manual frames list, e.g. '1,50,100,250' (only if --frames_source manual)")

    # Optional ROI metrics
    ap.add_argument("--enl_roi", default=None, help="ROI for ENL: y0,y1,x0,x1")
    ap.add_argument("--gcnr_in_roi", default=None, help="ROI inside lesion: y0,y1,x0,x1")
    ap.add_argument("--gcnr_out_roi", default=None, help="ROI outside/background: y0,y1,x0,x1")
    ap.add_argument("--gcnr_bins", type=int, default=256)

    args = ap.parse_args()

    dicom_path = Path(args.dicom)
    cine_run = Path(args.cine_run)
    full3d_run = Path(args.full3d_run)

    if args.out_dir is None:
        out_dir = cine_run / "compare_with_full3d"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    _, b01, _ = load_us_multiframe_as_float01(dicom_path)
    T, H, W = b01.shape

    cine_npy = cine_run / "denoised_cine.npy"
    cine_idx = cine_run / "frame_indices.npy"
    full_npy = full3d_run / "denoised_volume.npy"

    if not cine_npy.exists():
        raise FileNotFoundError(f"Missing: {cine_npy}")
    if not full_npy.exists():
        raise FileNotFoundError(f"Missing: {full_npy}")

    cine01 = np.load(cine_npy)  # (K,H,W)
    full01 = np.load(full_npy)  # expected (T,H,W)

    if full01.ndim != 3:
        raise ValueError(f"full3d denoised_volume.npy must be 3D (T,H,W). Got {full01.shape}")

    # Frame indices
    if args.frames_source == "cine":
        if not cine_idx.exists():
            raise FileNotFoundError(f"Missing: {cine_idx}")
        frames = np.load(cine_idx).astype(int).tolist()
    else:
        if args.frames is None:
            raise ValueError("Provide --frames when --frames_source manual")
        # parse "1,50,100"
        frames = []
        for tok in args.frames.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            t = int(tok)
            if t < 0:
                t = T + t
            if 0 <= t < T:
                frames.append(t)
        frames = sorted(set(frames))

    K = len(frames)
    if cine01.shape[0] != K:
        raise ValueError(
            f"cine output K mismatch: denoised_cine.npy has {cine01.shape[0]} frames "
            f"but frame list has {K}. (Check that cine_run matches the frame_indices.npy.)"
        )

    # Shape checks
    if cine01.shape[1:] != (H, W):
        raise ValueError(f"cine shape {cine01.shape} incompatible with DICOM (H,W)=({H},{W})")
    if full01.shape != (T, H, W):
        raise ValueError(f"full3d shape {full01.shape} incompatible with DICOM (T,H,W)=({T},{H},{W})")

    # Save per-frame panels
    panels_dir = out_dir / "panels"
    for i, t in enumerate(frames):
        save_panel(
            panels_dir / f"compare_{t:04d}.png",
            b2d=b01[t],
            cine2d=cine01[i],
            full2d=full01[t],
            t=t,
            show_diff=bool(args.show_diff),
        )

    # Save one grid
    save_grid(out_dir / "grid.png", frames, b01, cine01, full01, show_diff=bool(args.show_diff))

    # Optional metrics
    enl_roi = parse_roi(args.enl_roi)
    g_in = parse_roi(args.gcnr_in_roi)
    g_out = parse_roi(args.gcnr_out_roi)
    do_gcnr = (g_in is not None) or (g_out is not None)
    if do_gcnr and (g_in is None or g_out is None):
        raise ValueError("Provide BOTH --gcnr_in_roi and --gcnr_out_roi, or neither.")

    if enl_roi is not None or do_gcnr:
        csv_path = out_dir / "roi_metrics.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("frame,enl_b,enl_cine,enl_full3d,gcnr_b,gcnr_cine,gcnr_full3d\n")
            for i, t in enumerate(frames):
                b2d = b01[t]
                c2d = cine01[i]
                f2d = full01[t]

                e_b = enl(b2d, enl_roi) if enl_roi is not None else float("nan")
                e_c = enl(c2d, enl_roi) if enl_roi is not None else float("nan")
                e_f = enl(f2d, enl_roi) if enl_roi is not None else float("nan")

                g_b = gcnr(b2d, g_in, g_out, bins=int(args.gcnr_bins)) if do_gcnr else float("nan")
                g_c = gcnr(c2d, g_in, g_out, bins=int(args.gcnr_bins)) if do_gcnr else float("nan")
                g_f = gcnr(f2d, g_in, g_out, bins=int(args.gcnr_bins)) if do_gcnr else float("nan")

                def _fmt(x):
                    return "nan" if (x is None or not np.isfinite(x)) else f"{float(x):.6f}"

                f.write(f"{t},{_fmt(e_b)},{_fmt(e_c)},{_fmt(e_f)},{_fmt(g_b)},{_fmt(g_c)},{_fmt(g_f)}\n")

        print("Saved comparison panels to:", panels_dir.resolve())
        print("Saved grid to:", (out_dir / "grid.png").resolve())
        print("Saved ROI metrics to:", csv_path.resolve())
    else:
        print("Saved comparison panels to:", panels_dir.resolve())
        print("Saved grid to:", (out_dir / "grid.png").resolve())
        print("No ROI metrics requested (no --enl_roi and no gCNR ROIs).")


if __name__ == "__main__":
    main()
