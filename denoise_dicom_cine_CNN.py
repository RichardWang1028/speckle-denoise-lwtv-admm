# denoise_dicom_cine_CNN.py
# 2D (per-frame) ADMM denoising for multi-frame ultrasound DICOM
# with learned alpha-map (CNN) and ROI-based ENL / gCNR evaluation.
#
# Frame selection priority:
#   1) --frames "1,50,100,250,300,500" or "0:500:50" (inclusive end)
#   2) --sample_k K  (auto evenly spaced across detected T)
#   3) --start/--stop (consecutive range)
#
# Metrics:
#   --enl_roi "y0,y1,x0,x1"
#   --gcnr_in_roi "y0,y1,x0,x1" --gcnr_out_roi "y0,y1,x0,x1" --gcnr_bins 256
#
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import torch

# Your project modules
from alpha_net import AlphaUNetSmall, alpha_from_logits
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


def resolve_dicom_path(dicom_arg: str) -> Path:
    """
    Minimal resolver:
      - if dicom_arg exists, use it
      - else if dicom_arg + ".dcm" exists, use it
      - else raise
    """
    p = Path(dicom_arg)
    if p.exists():
        return p
    p2 = Path(dicom_arg + ".dcm")
    if p2.exists():
        return p2
    raise FileNotFoundError(f"DICOM not found: '{dicom_arg}' (also tried '{dicom_arg}.dcm')")


def load_us_multiframe_as_float01(
    dicom_path: str | Path, eps: float = 1e-6
) -> Tuple[pydicom.Dataset, np.ndarray, float]:
    """
    Returns (ds, b01, scale), where b01 is float32 in [eps,1] with shape (T,H,W).
    """
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array

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
    """
    Save a DICOM with NumberOfFrames = u01.shape[0].
    If you denoise a sparse subset of frames, the output DICOM contains ONLY those frames
    in the order processed. The mapping to original indices is saved separately.
    """
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
    ds.DerivationDescription = "Denoised with 2D ADMM per selected frame (LWTV-Log)."

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
# ROI parsing and metrics
# ============================================================
def parse_roi_2d(s: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse ROI as "y0,y1,x0,x1".
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
    y0c = max(0, min(H, y0))
    y1c = max(0, min(H, y1))
    x0c = max(0, min(W, x0))
    x1c = max(0, min(W, x1))
    if not (y0c < y1c and x0c < x1c):
        return np.empty((0,), dtype=np.float64)
    return img01[y0c:y1c, x0c:x1c].astype(np.float64).ravel()


def enl_on_roi(img01: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """
    ENL = (mu^2) / (sigma^2) on ROI, with sample sigma (ddof=1) to match thesis.
    """
    patch = _safe_crop(img01, roi)
    n = int(patch.size)
    if n < 2:
        return float("nan")
    mu = float(np.mean(patch))
    sigma = float(np.std(patch, ddof=1))
    if sigma <= 0:
        return float("nan")
    return float((mu * mu) / (sigma * sigma))


def gcnr_on_rois(img01: np.ndarray,
                 roi_in: Tuple[int, int, int, int],
                 roi_out: Tuple[int, int, int, int],
                 bins: int = 256) -> float:
    """
    gCNR = 1 - OVL, where OVL = ∫ min(p_in(t), p_out(t)) dt over t in [0,1].
    We approximate via histograms on [0,1] with density=True:
        OVL ≈ sum_i min(pdf_in[i], pdf_out[i]) * dt
    """
    pin = _safe_crop(img01, roi_in)
    pout = _safe_crop(img01, roi_out)
    if pin.size < 2 or pout.size < 2:
        return float("nan")

    pin = np.clip(pin, 0.0, 1.0)
    pout = np.clip(pout, 0.0, 1.0)

    pdf_in, edges = np.histogram(pin, bins=bins, range=(0.0, 1.0), density=True)
    pdf_out, _ = np.histogram(pout, bins=bins, range=(0.0, 1.0), density=True)
    dt = float(edges[1] - edges[0])

    ovl = float(np.sum(np.minimum(pdf_in, pdf_out)) * dt)
    gcnr = 1.0 - ovl
    return float(np.clip(gcnr, 0.0, 1.0))


# ============================================================
# Visualization
# ============================================================
def save_compare_png_frame(b2d01: np.ndarray, u2d01: np.ndarray, out_path: Path, t: int):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(b2d01, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Input b (frame {t})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(u2d01, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Denoised u (frame {t})")
    plt.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_preview_selected_frames(b01_full: np.ndarray, u_sel01: np.ndarray, out_dir: Path, frame_ids: List[int]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, t in enumerate(frame_ids):
        if not (0 <= t < b01_full.shape[0]):
            continue
        b = b01_full[t]
        u = u_sel01[j]
        Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"b_{t:04d}.png")
        Image.fromarray((np.clip(u, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"u_{t:04d}.png")


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


def _pad2d_to_mult(img: np.ndarray, mult: int = 16):
    H, W = img.shape
    Hp = ((H + mult - 1) // mult) * mult
    Wp = ((W + mult - 1) // mult) * mult
    ph, pw = Hp - H, Wp - W
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    imgp = np.pad(img, ((pt, pb), (pl, pr)), mode="reflect")
    return imgp, (pt, pb, pl, pr)


def _unpad2d(imgp: np.ndarray, pads):
    pt, pb, pl, pr = pads
    return imgp[pt : imgp.shape[0] - pb, pl : imgp.shape[1] - pr]


def infer_alpha_map_2d(net, b2d01: np.ndarray, device: torch.device, alpha_min: float, alpha_max: float) -> np.ndarray:
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

    if isinstance(alpha, (float, int)):
        alpha_arg = float(alpha)
    else:
        alpha_arg = backend.to_device(alpha.astype(np.float32))

    # support both solver signatures (with/without state)
    try:
        u_dev, state_out, _hist = admm_speckle_scaled(
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
        u_dev, state_out, _hist = admm_speckle_scaled(
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
# Frame selection helpers
# ============================================================
def parse_frames_list(s: Optional[str], T: int) -> Optional[List[int]]:
    """
    Parse:
      "1,50,100,250" or ranges like "0:500:50" (inclusive end).
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None

    frames: List[int] = []
    tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
    for tok in tokens:
        if ":" in tok:
            parts = [p.strip() for p in tok.split(":")]
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                step = 1
            elif len(parts) == 3:
                a, b, step = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                raise ValueError("Bad --frames token. Use 'a:b' or 'a:b:step' or comma-separated indices.")
            if step == 0:
                raise ValueError("Bad --frames token: step cannot be 0.")
            if (b - a) * step < 0:
                raise ValueError("Bad --frames token: step sign inconsistent with range direction.")
            frames.extend(list(range(a, b + (1 if step > 0 else -1), step)))
        else:
            frames.append(int(tok))

    out: List[int] = []
    for t in frames:
        if t < 0:
            t = T + t
        if 0 <= t < T:
            out.append(int(t))

    out = sorted(set(out))
    if len(out) == 0:
        raise ValueError(f"--frames parsed to an empty set (T={T}).")
    return out


def pick_evenly_spaced_frames(T: int, k: int, margin: int = 0) -> List[int]:
    if T <= 0:
        raise ValueError("Empty DICOM: no frames found.")
    if k <= 0:
        raise ValueError("--sample_k must be > 0.")

    a = max(0, margin)
    b = max(0, T - 1 - margin)
    if b < a:
        a, b = 0, T - 1

    k_eff = min(k, b - a + 1)
    idx = np.linspace(a, b, num=k_eff, dtype=int)
    idx = np.unique(idx)
    return idx.tolist()


# ============================================================
# CLI
# ============================================================
@dataclass
class CineConfig:
    dicom: str
    out_root: str
    tag: str
    prefer_gpu: bool

    mu: float
    beta0: float
    admm: int
    pd: int
    pd_tol: float
    admm_tol: float
    auto_beta: bool

    start: int
    stop: int
    save_every: int

    frames: Optional[str]
    sample_k: int
    sample_margin: int
    no_warm_start: bool

    alpha_mode: str
    ckpt: Optional[str]
    alpha_min: float
    alpha_max: float
    alpha_const: float

    enl_roi: Optional[str]
    gcnr_in_roi: Optional[str]
    gcnr_out_roi: Optional[str]
    gcnr_bins: int


def main():
    ap = argparse.ArgumentParser(
        "Denoise multi-frame ultrasound DICOM by slicing into 2D and running ADMM per selected frame."
    )
    ap.add_argument("--dicom", required=True, help="Path to multi-frame DICOM (or basename)")
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
    ap.add_argument("--save_every", type=int, default=10, help="Save compare images every N processed frames")

    ap.add_argument("--frames", default=None,
                    help="Explicit frame indices, e.g. '1,50,100,250' or '0:500:50' (inclusive end).")
    ap.add_argument("--sample_k", type=int, default=0,
                    help="If >0 and --frames not set: auto-select k evenly spaced frames across detected depth.")
    ap.add_argument("--sample_margin", type=int, default=0,
                    help="Exclude this many boundary frames at each end when auto-sampling.")
    ap.add_argument("--no_warm_start", action="store_true",
                    help="Disable warm-start across frames (recommended when frames are non-consecutive).")

    ap.add_argument("--alpha_mode", choices=["learned", "const"], default="learned")
    ap.add_argument("--ckpt", default=None, help="alpha_net checkpoint (required if alpha_mode=learned)")
    ap.add_argument("--alpha_min", type=float, default=0.2)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_const", type=float, default=0.5, help="used if alpha_mode=const")

    ap.add_argument("--enl_roi", default=None, help="Optional ROI for ENL: y0,y1,x0,x1")
    ap.add_argument("--gcnr_in_roi", default=None, help="ROI inside lesion for gCNR: y0,y1,x0,x1")
    ap.add_argument("--gcnr_out_roi", default=None, help="ROI outside lesion (background) for gCNR: y0,y1,x0,x1")
    ap.add_argument("--gcnr_bins", type=int, default=256, help="Histogram bins for gCNR (default: 256)")

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
        frames=None if args.frames is None else str(args.frames),
        sample_k=int(args.sample_k),
        sample_margin=int(args.sample_margin),
        no_warm_start=bool(args.no_warm_start),
        alpha_mode=str(args.alpha_mode),
        ckpt=None if args.ckpt is None else str(args.ckpt),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        alpha_const=float(args.alpha_const),
        enl_roi=None if args.enl_roi is None else str(args.enl_roi),
        gcnr_in_roi=None if args.gcnr_in_roi is None else str(args.gcnr_in_roi),
        gcnr_out_roi=None if args.gcnr_out_roi is None else str(args.gcnr_out_roi),
        gcnr_bins=int(args.gcnr_bins),
    )

    # Resolve + load DICOM
    dicom_path = resolve_dicom_path(cfg.dicom)
    ds, b01, scale = load_us_multiframe_as_float01(dicom_path)
    T = int(b01.shape[0])

    # Frame selection
    frames = parse_frames_list(cfg.frames, T)
    if frames is None and cfg.sample_k > 0:
        frames = pick_evenly_spaced_frames(T, k=cfg.sample_k, margin=cfg.sample_margin)
    if frames is None:
        start = max(0, int(cfg.start))
        stop = T if int(cfg.stop) < 0 else min(T, int(cfg.stop))
        if not (start < stop):
            raise ValueError(f"Invalid frame range: start={start}, stop={stop}, T={T}")
        frames = list(range(start, stop))
    else:
        start = int(frames[0])
        stop = int(frames[-1]) + 1  # for logging only

    # Run directory
    run_tag = f"{dicom_path.name}_{cfg.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.out_root) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save mapping to original indices
    np.save(run_dir / "frame_indices.npy", np.array(frames, dtype=np.int32))
    (run_dir / "frame_indices.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")

    with tee_console_to_file(run_dir / "run.log"):
        # save config
        print("=== Config ===")
        print(json.dumps(asdict(cfg), indent=2))
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        print("\nLoaded DICOM:", dicom_path.name, "shape:", b01.shape, "dtype:", ds.pixel_array.dtype)
        print("Scale:", scale)
        print(f"Total frames T={T}")
        print(f"Selected {len(frames)} frames. (Logging range [{start}:{stop}) for convenience.)")
        if len(frames) <= 60:
            print("Selected frame indices:", frames)
        else:
            print("Selected frame indices head:", frames[:10], "... tail:", frames[-10:])

        # Backend
        backend = Backend(prefer_gpu=cfg.prefer_gpu, dtype=np.float32)
        backend.print_device_info()
        print("backend.on_gpu =", backend.on_gpu)

        # CNN alpha-net
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = None
        if cfg.alpha_mode == "learned":
            if cfg.ckpt is None:
                raise ValueError("--ckpt is required when --alpha_mode learned")
            net = load_alpha_net(cfg.ckpt, device)
            print("alpha_net loaded on:", device)

        # Parse ROIs
        enl_roi_parsed = parse_roi_2d(cfg.enl_roi)
        gcnr_in_roi_parsed = parse_roi_2d(cfg.gcnr_in_roi)
        gcnr_out_roi_parsed = parse_roi_2d(cfg.gcnr_out_roi)
        if (gcnr_in_roi_parsed is None) ^ (gcnr_out_roi_parsed is None):
            raise ValueError("Provide BOTH --gcnr_in_roi and --gcnr_out_roi, or neither.")

        # Output stack aligns with frames list order
        out = np.empty((len(frames), b01.shape[1], b01.shape[2]), dtype=np.float32)

        # metrics log: per frame
        metrics_rows = []  # dicts: frame,enl_in,enl_out,gcnr_in,gcnr_out

        state = None
        t0_all = time.time()

        for idx, t in enumerate(frames):
            # warm-start handling
            if cfg.no_warm_start:
                state = None
            elif idx > 0 and (frames[idx] != frames[idx - 1] + 1):
                state = None

            b2d = b01[t]

            # alpha
            if cfg.alpha_mode == "const":
                alpha_used = float(cfg.alpha_const)
            else:
                assert net is not None
                alpha_used = infer_alpha_map_2d(net, b2d, device, cfg.alpha_min, cfg.alpha_max)

            # ADMM
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

            # Metrics
            row = {"frame": int(t), "enl_in": np.nan, "enl_out": np.nan, "gcnr_in": np.nan, "gcnr_out": np.nan}
            if enl_roi_parsed is not None:
                row["enl_in"] = float(enl_on_roi(b2d, enl_roi_parsed))
                row["enl_out"] = float(enl_on_roi(u2d, enl_roi_parsed))

            if gcnr_in_roi_parsed is not None:
                row["gcnr_in"] = float(gcnr_on_rois(b2d, gcnr_in_roi_parsed, gcnr_out_roi_parsed, bins=cfg.gcnr_bins))
                row["gcnr_out"] = float(gcnr_on_rois(u2d, gcnr_in_roi_parsed, gcnr_out_roi_parsed, bins=cfg.gcnr_bins))

            metrics_rows.append(row)

            # progress
            if idx % 10 == 0:
                print(f"Done frame {t} ({idx+1}/{len(frames)})")

            # comparisons
            if cfg.save_every > 0 and (idx % cfg.save_every) == 0:
                save_compare_png_frame(
                    b2d01=b2d,
                    u2d01=u2d,
                    out_path=run_dir / "compare_frames" / f"compare_{t:04d}.png",
                    t=t,
                )

        total_sec = time.time() - t0_all
        print(f"Total denoise time: {total_sec:.2f} sec for {len(frames)} frames.")

        # Save arrays
        np.save(run_dir / "denoised_cine.npy", out)
        print("Saved:", (run_dir / "denoised_cine.npy").resolve())

        # Save DICOM (selected frames only)
        out_dicom = run_dir / f"{dicom_path.name}_denoised_cine2d.dcm"
        saved = save_multiframe_dicom_like_input(ds, out, out_dicom, scale=scale)
        print("Saved denoised DICOM:", saved.resolve())

        # Preview PNGs
        prev_dir = run_dir / "preview_frames"
        if len(frames) <= 12:
            preview_pos = list(range(len(frames)))
        else:
            preview_pos = sorted(set([0, len(frames)//4, len(frames)//2, (3*len(frames))//4, len(frames)-1]))
        preview_ids = [frames[p] for p in preview_pos]
        preview_u = out[preview_pos]
        save_preview_selected_frames(b01, preview_u, prev_dir, preview_ids)
        print("Saved preview PNGs to:", prev_dir.resolve())

        # Save metrics (CSV + summary)
        if (enl_roi_parsed is not None) or (gcnr_in_roi_parsed is not None):
            metrics_csv = run_dir / "roi_metrics.csv"
            with open(metrics_csv, "w", encoding="utf-8") as f:
                f.write("frame,enl_in,enl_out,gcnr_in,gcnr_out\n")
                for r in metrics_rows:
                    f.write(
                        f"{r['frame']},"
                        f"{r['enl_in']:.6f}," if np.isfinite(r["enl_in"]) else f"{r['frame']},nan,"
                    )
                    # continue line carefully
            # re-write properly (avoid partial write above)
            with open(metrics_csv, "w", encoding="utf-8") as f:
                f.write("frame,enl_in,enl_out,gcnr_in,gcnr_out\n")
                for r in metrics_rows:
                    def _fmt(x):
                        return "nan" if (x is None or not np.isfinite(x)) else f"{float(x):.6f}"
                    f.write(f"{r['frame']},{_fmt(r['enl_in'])},{_fmt(r['enl_out'])},{_fmt(r['gcnr_in'])},{_fmt(r['gcnr_out'])}\n")

            print("Saved ROI metrics to:", metrics_csv.resolve())

            # Summary statistics over finite values
            def finite_mean_std(vals: List[float]):
                a = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
                if a.size == 0:
                    return {"mean": None, "std": None, "n": 0}
                return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size >= 2 else 0.0, "n": int(a.size)}

            enl_in_list = [r["enl_in"] for r in metrics_rows]
            enl_out_list = [r["enl_out"] for r in metrics_rows]
            g_in_list = [r["gcnr_in"] for r in metrics_rows]
            g_out_list = [r["gcnr_out"] for r in metrics_rows]

            summary = {
                "frames_used": frames,
                "ENL_in": finite_mean_std(enl_in_list),
                "ENL_out": finite_mean_std(enl_out_list),
                "gCNR_in": finite_mean_std(g_in_list),
                "gCNR_out": finite_mean_std(g_out_list),
                "roi": {
                    "enl_roi": enl_roi_parsed,
                    "gcnr_in_roi": gcnr_in_roi_parsed,
                    "gcnr_out_roi": gcnr_out_roi_parsed,
                    "gcnr_bins": cfg.gcnr_bins,
                },
            }
            (run_dir / "roi_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("Saved ROI metrics summary to:", (run_dir / "roi_metrics_summary.json").resolve())


if __name__ == "__main__":
    main()
