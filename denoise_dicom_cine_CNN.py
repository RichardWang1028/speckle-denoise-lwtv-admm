# denoise_dicom_cine.py
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

import torch

# CNN alpha components
from alpha_net import AlphaUNetSmall, alpha_from_logits

# 2D ADMM solver
from solver_admm_2d import Backend, admm_speckle_scaled


# ============================================================
# Logging helper
# ============================================================
@contextmanager
def tee_console_to_file(log_path: Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")

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

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, f)
    sys.stderr = Tee(old_err, f)
    try:
        print("========== RUN LOG ==========")
        print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Log file:", log_path.resolve())
        print("=============================\n")
        yield
    finally:
        print("\n=============================")
        print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=============================")
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()


# ============================================================
# DICOM I/O
# ============================================================
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


def load_us_multiframe_as_float01(dicom_path: str | Path, eps: float = 1e-6) -> Tuple[pydicom.Dataset, np.ndarray, float]:
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array

    # Ensure shape (T,H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D pixel_array, got shape {arr.shape}")

    scale = _infer_max_value(ds, arr)
    b01 = arr.astype(np.float32) / float(scale)
    b01 = np.clip(b01, eps, 1.0)  # positivity for log-term
    return ds, b01, float(scale)


def save_multiframe_dicom_like_input(
    ds_in: pydicom.Dataset, u01: np.ndarray, out_path: str | Path, scale: float
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u01 = np.asarray(u01, dtype=np.float32)
    u01 = np.clip(u01, 0.0, 1.0)
    u_int = np.round(u01 * float(scale))

    in_dtype = ds_in.pixel_array.dtype
    if in_dtype == np.uint16:
        u_int = np.clip(u_int, 0, int(scale)).astype(np.uint16)
    else:
        u_int = np.clip(u_int, 0, 255).astype(np.uint8)

    ds = ds_in.copy()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.DerivationDescription = "Denoised with 2D ADMM per frame (LWTV-Log)."

    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    if u_int.ndim == 2:
        u_int = u_int[None, ...]
    ds.NumberOfFrames = int(u_int.shape[0])
    ds.Rows = int(u_int.shape[1])
    ds.Columns = int(u_int.shape[2])
    ds.PixelData = u_int.tobytes(order="C")

    pydicom.dcmwrite(str(out_path), ds, write_like_original=False)
    return out_path


# ============================================================
# Metrics / no-GT indicators (optional)
# ============================================================
def enl_roi(img01: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """ENL = mean^2 / var on ROI. roi=(y0,y1,x0,x1)."""
    y0, y1, x0, x1 = roi
    patch = img01[y0:y1, x0:x1].astype(np.float64)
    m = float(np.mean(patch))
    v = float(np.var(patch))
    return (m * m) / (v + 1e-12)


# ============================================================
# Visualization
# ============================================================
def save_preview_frames(b01: np.ndarray, u01: np.ndarray, out_dir: Path, frame_ids: List[int]):
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in frame_ids:
        if not (0 <= t < b01.shape[0]):
            continue
        Image.fromarray((np.clip(b01[t], 0, 1) * 255).astype(np.uint8)).save(out_dir / f"b_{t:04d}.png")
        Image.fromarray((np.clip(u01[t], 0, 1) * 255).astype(np.uint8)).save(out_dir / f"u_{t:04d}.png")


def save_compare_png(b01: np.ndarray, u01: np.ndarray, out_path: Path, t: int, vmax: float = 1.0):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(b01[t], cmap="gray", vmin=0, vmax=vmax)
    plt.title(f"Input b (frame {t})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(u01[t], cmap="gray", vmin=0, vmax=vmax)
    plt.title(f"Denoised u (frame {t})")
    plt.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CNN alpha loader/infer
# ============================================================
def load_alpha_net(ckpt: str, device: torch.device) -> AlphaUNetSmall:
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"alpha_net checkpoint not found: {ckpt_path}")
    net = AlphaUNetSmall().to(device)
    net.eval()
    obj = torch.load(str(ckpt_path), map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        net.load_state_dict(obj["model_state_dict"], strict=True)
    elif isinstance(obj, dict):
        net.load_state_dict(obj, strict=False)
    else:
        raise RuntimeError("Unexpected checkpoint format for alpha_net.")
    return net


def _pad2d_to_mult(img, mult=16):
    H, W = img.shape
    Hp = ((H + mult - 1) // mult) * mult
    Wp = ((W + mult - 1) // mult) * mult
    ph, pw = Hp - H, Wp - W
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    imgp = np.pad(img, ((pt, pb), (pl, pr)), mode="reflect")
    return imgp, (pt, pb, pl, pr)

def _unpad2d(imgp, pads):
    pt, pb, pl, pr = pads
    return imgp[pt:imgp.shape[0]-pb, pl:imgp.shape[1]-pr]

def infer_alpha_map_2d(net, b2d01, device, alpha_min, alpha_max):
    bpad, pads = _pad2d_to_mult(b2d01, mult=16)
    x = torch.from_numpy(bpad[None, None, ...]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = net(x)
        a = alpha_from_logits(logits, alpha_min, alpha_max)[0, 0].detach().cpu().numpy().astype(np.float32)
    return _unpad2d(a, pads)



# ============================================================
# ADMM per-frame runner
# ============================================================
def admm_frame(
    backend: Backend,
    b2d01: np.ndarray,
    alpha: np.ndarray | float,
    mu: float,
    beta0: float,
    n_admm_iters: int,
    n_pd_iters: int,
    state: Any,
    pd_tol: float,
    admm_tol: float,
    auto_beta: bool,
) -> Tuple[np.ndarray, Any]:
    b_dev = backend.to_device(b2d01.astype(np.float32))

    # alpha to device if map
    if isinstance(alpha, (float, int)):
        alpha_arg = float(alpha)
    else:
        alpha_arg = backend.to_device(alpha.astype(np.float32))

    # call with/without state depending on solver signature
    try:
        u_dev, state_out, hist = admm_speckle_scaled(
            backend,
            b=b_dev,
            alpha=alpha_arg,
            mu=float(mu),
            beta=float(beta0),
            n_admm_iters=int(n_admm_iters),
            n_pd_iters=int(n_pd_iters),
            pd_tol=float(pd_tol),
            admm_tol=float(admm_tol),
            auto_beta=bool(auto_beta),
            verbose=False,
            state=state,
        )
    except TypeError:
        u_dev, state_out, hist = admm_speckle_scaled(
            backend,
            b=b_dev,
            alpha=alpha_arg,
            mu=float(mu),
            beta=float(beta0),
            n_admm_iters=int(n_admm_iters),
            n_pd_iters=int(n_pd_iters),
            pd_tol=float(pd_tol),
            admm_tol=float(admm_tol),
            auto_beta=bool(auto_beta),
            verbose=False,
        )

    u2d = backend.to_cpu(u_dev).astype(np.float32)
    u2d = np.clip(u2d, 0.0, 1.0)
    return u2d, state_out


# ============================================================
# CLI
# ============================================================
@dataclass
class CineConfig:
    dicom: str
    out_root: str
    tag: str
    prefer_gpu: bool

    # ADMM
    mu: float
    beta0: float
    admm: int
    pd: int
    pd_tol: float
    admm_tol: float
    auto_beta: bool

    # frames
    start: int
    stop: int
    save_every: int

    # alpha
    alpha_mode: str          # "learned" or "const"
    ckpt: Optional[str]
    alpha_min: float
    alpha_max: float
    alpha_const: float

    # ENL (optional)
    enl_roi: Optional[str]   # "y0,y1,x0,x1" or None


def parse_roi(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if s is None:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'y0,y1,x0,x1'")
    return parts[0], parts[1], parts[2], parts[3]


def main():
    ap = argparse.ArgumentParser("Denoise multi-frame ultrasound DICOM by slicing into 2D and running ADMM per frame.")
    ap.add_argument("--dicom", required=True, help="Path to multi-frame DICOM")
    ap.add_argument("--out_root", default="runs", help="Output root directory")
    ap.add_argument("--tag", default="us_cine2d", help="Run tag prefix")
    ap.add_argument("--prefer_gpu", action="store_true", help="Use CuPy backend for ADMM if available")

    ap.add_argument("--mu", type=float, default=4.0)
    ap.add_argument("--beta0", type=float, default=700.0)
    ap.add_argument("--admm", type=int, default=150)
    ap.add_argument("--pd", type=int, default=150)
    ap.add_argument("--pd_tol", type=float, default=1e-4)
    ap.add_argument("--admm_tol", type=float, default=1e-4)
    ap.add_argument("--auto_beta", action="store_true")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=-1, help="Stop frame (exclusive). -1 means all frames.")
    ap.add_argument("--save_every", type=int, default=10, help="Save compare images every N frames")

    ap.add_argument("--alpha_mode", choices=["learned", "const"], default="learned")
    ap.add_argument("--ckpt", default=None, help="alpha_net.pt (required if alpha_mode=learned)")
    ap.add_argument("--alpha_min", type=float, default=0.2)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_const", type=float, default=0.5, help="used if alpha_mode=const")

    ap.add_argument("--enl_roi", default=None, help="Optional ROI for ENL: y0,y1,x0,x1")

    args = ap.parse_args()

    cfg = CineConfig(
        dicom=str(args.dicom),
        out_root=str(args.out_root),
        tag=str(args.tag),
        prefer_gpu=bool(args.prefer_gpu),
        mu=float(args.mu),
        beta0=float(args.beta0),
        admm=int(args.admm),
        pd=int(args.pd),
        pd_tol=float(args.pd_tol),
        admm_tol=float(args.admm_tol),
        auto_beta=bool(args.auto_beta),
        start=int(args.start),
        stop=int(args.stop),
        save_every=int(args.save_every),
        alpha_mode=str(args.alpha_mode),
        ckpt=None if args.ckpt is None else str(args.ckpt),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        alpha_const=float(args.alpha_const),
        enl_roi=None if args.enl_roi is None else str(args.enl_roi),
    )

    dicom_path = Path(cfg.dicom)
    ds, b01, scale = load_us_multiframe_as_float01(dicom_path)
    T = b01.shape[0]

    start = max(0, cfg.start)
    stop = T if cfg.stop < 0 else min(T, cfg.stop)
    if not (start < stop):
        raise ValueError(f"Invalid frame range: start={start}, stop={stop}, T={T}")

    run_tag = f"{dicom_path.name}_{cfg.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.out_root) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_console_to_file(run_dir / "run.log"):
        print("=== Config ===")
        print(json.dumps(asdict(cfg), indent=2))
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        print("Loaded DICOM:", dicom_path.name, "shape:", b01.shape, "dtype:", ds.pixel_array.dtype)
        print("Scale:", scale)
        print(f"Processing frames [{start}:{stop}) out of T={T}")

        # Backend for ADMM
        backend = Backend(prefer_gpu=cfg.prefer_gpu, dtype=np.float32)
        backend.print_device_info()
        print("backend.on_gpu =", backend.on_gpu)

        # CNN alpha-net (optional)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = None
        if cfg.alpha_mode == "learned":
            if cfg.ckpt is None:
                raise ValueError("--ckpt is required when --alpha_mode learned")
            net = load_alpha_net(cfg.ckpt, device)
            print("alpha_net loaded on:", device)

        out = np.empty((stop - start, b01.shape[1], b01.shape[2]), dtype=np.float32)
        state = None  # warm start across frames

        roi = parse_roi(cfg.enl_roi)
        enl_log = []

        t0_all = time.time()
        for idx, t in enumerate(range(start, stop)):
            b2d = b01[t]

            if cfg.alpha_mode == "const":
                alpha_used = float(cfg.alpha_const)
            else:
                assert net is not None
                alpha_used = infer_alpha_map_2d(net, b2d, device, cfg.alpha_min, cfg.alpha_max)

            u2d, state = admm_frame(
                backend=backend,
                b2d01=b2d,
                alpha=alpha_used,
                mu=cfg.mu,
                beta0=cfg.beta0,
                n_admm_iters=cfg.admm,
                n_pd_iters=cfg.pd,
                state=state,
                pd_tol=cfg.pd_tol,
                admm_tol=cfg.admm_tol,
                auto_beta=cfg.auto_beta,
            )

            out[idx] = u2d

            if roi is not None:
                enl_in = enl_roi(b2d, roi)
                enl_out = enl_roi(u2d, roi)
                enl_log.append((t, float(enl_in), float(enl_out)))

            if (idx % 10) == 0:
                print(f"Done frame {t} ({idx+1}/{stop-start})")

            if cfg.save_every > 0 and (idx % cfg.save_every) == 0:
                save_compare_png(b01=b01, u01=np.concatenate([b01[:start], out, b01[stop:]], axis=0),  # hack to index t
                                 out_path=run_dir / "compare_frames" / f"compare_{t:04d}.png", t=t)

        total_sec = time.time() - t0_all
        print(f"Total denoise time: {total_sec:.2f} sec for {stop-start} frames.")

        # Save numpy
        np.save(run_dir / "denoised_cine.npy", out)
        print("Saved:", (run_dir / "denoised_cine.npy").resolve())

        # Save DICOM (same #frames as processed range)
        out_dicom = run_dir / f"{dicom_path.name}_denoised_cine2d.dcm"
        saved = save_multiframe_dicom_like_input(ds, out, out_dicom, scale=scale)
        print("Saved denoised DICOM:", saved.resolve())

        # Previews
        prev_dir = run_dir / "preview_frames"
        preview_ids = [start, min(start + 10, stop - 1), (start + stop) // 2, stop - 1]
        preview_ids = sorted(set([i for i in preview_ids if start <= i < stop]))
        save_preview_frames(b01[start:stop], out, prev_dir, [i - start for i in preview_ids])
        print("Saved preview PNGs to:", prev_dir.resolve())

        # ENL log if requested
        if roi is not None:
            enl_path = run_dir / "enl.csv"
            with open(enl_path, "w", encoding="utf-8") as f:
                f.write("frame,enl_in,enl_out\n")
                for t, ei, eo in enl_log:
                    f.write(f"{t},{ei:.6f},{eo:.6f}\n")
            print("Saved ENL to:", enl_path.resolve())


if __name__ == "__main__":
    main()
