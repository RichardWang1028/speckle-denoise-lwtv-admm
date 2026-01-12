# denoise_dicom_full3d.py
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

import torch
from alpha_net import AlphaUNetSmall, alpha_from_logits

from solver_admm_3d import Backend, admm_speckle_scaled_3d


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
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (Z,H,W), got {arr.shape}")

    scale = _infer_max_value(ds, arr)
    b01 = arr.astype(np.float32) / float(scale)
    b01 = np.clip(b01, eps, 1.0)
    return ds, b01, float(scale)


def save_denoised_dicom(ds_in: pydicom.Dataset, u01: np.ndarray, out_path: str | Path, scale: float) -> Path:
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
    ds.DerivationDescription = "Denoised with full 3D ADMM (LWTV-Log + TV3D)."

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


def save_slice_comparisons(b01: np.ndarray, u01: np.ndarray, out_dir: Path, slice_ids=None, vmax: float = 1.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Z = b01.shape[0]
    if slice_ids is None:
        slice_ids = [0, Z // 4, Z // 2, (3 * Z) // 4, Z - 1]
        slice_ids = sorted(set(int(i) for i in slice_ids if 0 <= i < Z))

    for z in slice_ids:
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(b01[z], cmap="gray", vmin=0, vmax=vmax)
        plt.title(f"Input b (slice {z})")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(u01[z], cmap="gray", vmin=0, vmax=vmax)
        plt.title(f"Denoised u (slice {z})")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"compare_slice_{z:04d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


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

def _pad_hw_to_mult(vol_zhw: np.ndarray, mult: int = 32):
    """
    Pad a (Z,H,W) volume in (H,W) to the next multiple of `mult` using reflect padding.
    Returns padded volume and pads (pt,pb,pl,pr).
    """
    Z, H, W = vol_zhw.shape
    Hp = ((H + mult - 1) // mult) * mult
    Wp = ((W + mult - 1) // mult) * mult

    ph, pw = Hp - H, Wp - W
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2

    vol_pad = np.pad(vol_zhw, ((0, 0), (pt, pb), (pl, pr)), mode="reflect")
    return vol_pad, (pt, pb, pl, pr)

def _unpad_hw(vol_zhw_pad: np.ndarray, pads):
    """Crop a padded (Z,Hp,Wp) volume back to original (Z,H,W)."""
    pt, pb, pl, pr = pads
    Z, Hp, Wp = vol_zhw_pad.shape
    return vol_zhw_pad[:, pt:Hp - pb, pl:Wp - pr]



def infer_alpha_volume_by_slices(
    net: AlphaUNetSmall,
    b01: np.ndarray,  # (Z,H,W)
    device: torch.device,
    alpha_min: float,
    alpha_max: float,
    infer_batch: int = 1,
) -> np.ndarray:
    Z, H, W = b01.shape

    # --- PAD (H,W) to avoid UNet skip-connection mismatch ---
    b01_pad, pads = _pad_hw_to_mult(b01, mult=32)  # mult=32 is safe for typical U-Net depths
    _, Hp, Wp = b01_pad.shape

    out_pad = np.empty((Z, Hp, Wp), dtype=np.float32)

    infer_batch = max(1, int(infer_batch))
    for z0 in range(0, Z, infer_batch):
        z1 = min(Z, z0 + infer_batch)

        x = torch.from_numpy(b01_pad[z0:z1, None, ...]).to(device=device, dtype=torch.float32)  # (B,1,Hp,Wp)
        with torch.no_grad():
            logits = net(x)
            a = alpha_from_logits(logits, alpha_min, alpha_max)[:, 0]  # (B,Hp,Wp)

        out_pad[z0:z1] = a.detach().cpu().numpy().astype(np.float32)

        if (z0 // infer_batch) % 10 == 0:
            print(f"alpha slice batch {z0}:{z1} / {Z}")

    # --- CROP back to original (H,W) ---
    out = _unpad_hw(out_pad, pads)  # (Z,H,W)
    return out


# ============================================================
# Slice selection helpers (for qualitative comparisons)
# ============================================================
def parse_slices_list(s: Optional[str], Z: int) -> Optional[list[int]]:
    """
    Parse a user-specified slice list.

    Examples:
      --slices "0,50,100,250,300,500"
      --slices "0:500:50"   (inclusive)
      --slices "-1"         (last slice)

    Returns sorted unique indices, or None if s is None/empty.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None

    idxs: list[int] = []
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
                raise ValueError("Bad --slices token. Use 'a:b' or 'a:b:step' or comma-separated indices.")
            if step == 0:
                raise ValueError("Bad --slices token: step cannot be 0.")
            if (b - a) * step < 0:
                raise ValueError("Bad --slices token: step sign is inconsistent with range direction.")
            idxs.extend(list(range(a, b + (1 if step > 0 else -1), step)))
        else:
            idxs.append(int(tok))

    out: list[int] = []
    for z in idxs:
        if z < 0:
            z = Z + z
        if 0 <= z < Z:
            out.append(int(z))

    out = sorted(set(out))
    if len(out) == 0:
        raise ValueError(f"--slices parsed to an empty set (Z={Z}). Check your indices.")
    return out


def pick_evenly_spaced_slices(Z: int, k: int, margin: int = 0) -> list[int]:
    Z = int(Z)
    k = int(k)
    margin = int(margin)
    if Z <= 0:
        raise ValueError("Empty DICOM: no slices found.")
    if k <= 0:
        raise ValueError("sample_k must be > 0.")

    a = max(0, margin)
    b = max(0, Z - 1 - margin)
    if b < a:
        a, b = 0, Z - 1

    idx = np.linspace(a, b, num=min(k, b - a + 1), dtype=int)
    idx = np.unique(idx)
    return idx.tolist()


@dataclass
class Full3DConfig:
    dicom: str
    out_root: str
    tag: str
    prefer_gpu: bool

    alpha_source: str       # "const" or "slice_cnn"
    ckpt: Optional[str]
    alpha_min: float
    alpha_max: float
    alpha_const: float
    infer_batch: int

    mu: float
    beta0: float
    admm: int
    pd: int
    pd_tol: float
    admm_tol: float
    auto_beta: bool
    z_limit: int

    # slice selection for saved PNG comparisons (does not change the 3D solve)
    slices: Optional[str]
    sample_k: int
    sample_margin: int


def main():
    ap = argparse.ArgumentParser("Full 3D ADMM denoising for multi-frame ultrasound DICOM.")
    ap.add_argument("--dicom", required=True)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--tag", default="us_full3d")
    ap.add_argument("--prefer_gpu", action="store_true")

    ap.add_argument("--alpha_source", choices=["const", "slice_cnn"], default="const")
    ap.add_argument("--ckpt", default=None, help="alpha_net.pt (required if alpha_source=slice_cnn)")
    ap.add_argument("--alpha_min", type=float, default=0.2)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_const", type=float, default=0.5)
    ap.add_argument("--infer_batch", type=int, default=1)

    ap.add_argument("--mu", type=float, default=4.0)
    ap.add_argument("--beta0", type=float, default=700.0)
    ap.add_argument("--admm", type=int, default=120)
    ap.add_argument("--pd", type=int, default=80)
    ap.add_argument("--pd_tol", type=float, default=1e-4)
    ap.add_argument("--admm_tol", type=float, default=1e-4)
    ap.add_argument("--auto_beta", action="store_true")

    ap.add_argument("--z_limit", type=int, default=-1, help="-1 for full volume, else use first z_limit slices")

    # Slice selection for qualitative comparisons
    ap.add_argument("--slices", default=None,
                    help="Explicit slice indices for saved PNG comparisons, e.g. \'0,50,100\' or \'0:500:50\'.")
    ap.add_argument("--sample_k", type=int, default=6,
                    help="If --slices is not given: save k evenly spaced slices for comparisons.")
    ap.add_argument("--sample_margin", type=int, default=0,
                    help="Exclude this many boundary slices at each end when auto-sampling.")
    args = ap.parse_args()

    cfg = Full3DConfig(
        dicom=str(args.dicom),
        out_root=str(args.out_root),
        tag=str(args.tag),
        prefer_gpu=bool(args.prefer_gpu),
        alpha_source=str(args.alpha_source),
        ckpt=None if args.ckpt is None else str(args.ckpt),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        alpha_const=float(args.alpha_const),
        infer_batch=int(args.infer_batch),
        mu=float(args.mu),
        beta0=float(args.beta0),
        admm=int(args.admm),
        pd=int(args.pd),
        pd_tol=float(args.pd_tol),
        admm_tol=float(args.admm_tol),
        auto_beta=bool(args.auto_beta),
        z_limit=int(args.z_limit),
        slices=None if args.slices is None else str(args.slices),
        sample_k=int(args.sample_k),
        sample_margin=int(args.sample_margin),
    )

    dicom_path = Path(cfg.dicom)
    run_tag = f"{dicom_path.name}_{cfg.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.out_root) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_console_to_file(run_dir / "run.log"):
        print("=== Config ===")
        print(json.dumps(asdict(cfg), indent=2))
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        ds, b01, scale = load_us_multiframe_as_float01(dicom_path)
        print("Loaded:", dicom_path.name, "shape:", b01.shape, "dtype:", ds.pixel_array.dtype)
        print("Scale:", scale)

        # Decide which slices to export as qualitative comparisons
        Z = int(b01.shape[0])
        slice_ids = parse_slices_list(cfg.slices, Z)
        if slice_ids is None:
            k = int(cfg.sample_k) if int(cfg.sample_k) > 0 else 6
            slice_ids = pick_evenly_spaced_slices(Z, k=k, margin=int(cfg.sample_margin))
        # Note: if z_limit is applied below, slice_ids will still be valid because Z is recomputed after truncation.
        
        if cfg.z_limit > 0:
            b01 = b01[: cfg.z_limit]
            print("Using first z_limit slices:", b01.shape)
            Z = int(b01.shape[0])
            # Clamp previously chosen slice_ids to the truncated volume
            slice_ids = [z for z in slice_ids if 0 <= z < Z]
            if len(slice_ids) == 0:
                slice_ids = pick_evenly_spaced_slices(Z, k=max(1, min(int(cfg.sample_k), Z)), margin=int(cfg.sample_margin))
            np.save(run_dir / "slice_indices.npy", np.array(slice_ids, dtype=np.int32))
            (run_dir / "slice_indices.json").write_text(json.dumps(slice_ids, indent=2), encoding="utf-8")

        # Decide which slices to export as qualitative comparisons
        Z = int(b01.shape[0])
        slice_ids = parse_slices_list(cfg.slices, Z)
        if slice_ids is None:
            k = int(cfg.sample_k) if int(cfg.sample_k) > 0 else 6
            slice_ids = pick_evenly_spaced_slices(Z, k=k, margin=int(cfg.sample_margin))
        print(f"Selected {len(slice_ids)} slice(s) for comparisons: {slice_ids}")
        np.save(run_dir / "slice_indices.npy", np.array(slice_ids, dtype=np.int32))
        (run_dir / "slice_indices.json").write_text(json.dumps(slice_ids, indent=2), encoding="utf-8")

        # alpha selection
        alpha_for_solver: Any = float(cfg.alpha_const)
        alpha_vol: Optional[np.ndarray] = None

        if cfg.alpha_source == "slice_cnn":
            if cfg.ckpt is None:
                raise ValueError("--ckpt is required when --alpha_source slice_cnn")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = load_alpha_net(cfg.ckpt, device)
            print("alpha_net loaded on:", device)

            alpha_vol = infer_alpha_volume_by_slices(
                net=net,
                b01=b01,
                device=device,
                alpha_min=cfg.alpha_min,
                alpha_max=cfg.alpha_max,
                infer_batch=cfg.infer_batch,
            )
            np.save(run_dir / "alpha_volume.npy", alpha_vol)
            print("Saved alpha volume:", (run_dir / "alpha_volume.npy").resolve())

            # Try to pass alpha volume into solver; if not supported, fall back to scalar mean.
            alpha_for_solver = alpha_vol  # will attempt below

        backend = Backend(prefer_gpu=cfg.prefer_gpu, dtype=np.float32)
        backend.print_device_info()
        print("backend.on_gpu:", backend.on_gpu)

        b_dev = backend.to_device(b01.astype(np.float32))

        # Move alpha to device if it is a volume
        alpha_arg = alpha_for_solver
        if isinstance(alpha_for_solver, np.ndarray):
            try:
                alpha_arg = backend.to_device(alpha_for_solver.astype(np.float32))
            except Exception:
                alpha_arg = float(np.mean(alpha_for_solver))
                print("[WARN] Could not move alpha volume to device; using mean(alpha) =", alpha_arg)

        # Run full 3D ADMM
        t0 = time.time()
        try:
            u_dev, state, hist = admm_speckle_scaled_3d(
                backend,
                b=b_dev,
                alpha=alpha_arg,
                mu=float(cfg.mu),
                beta=float(cfg.beta0),
                n_admm_iters=int(cfg.admm),
                n_pd_iters=int(cfg.pd),
                pd_tol=float(cfg.pd_tol),
                admm_tol=float(cfg.admm_tol),
                auto_beta=bool(cfg.auto_beta),
                verbose=True,
            )
        except TypeError:
            # fallback: some implementations only accept scalar alpha
            if isinstance(alpha_for_solver, np.ndarray):
                alpha_mean = float(np.mean(alpha_for_solver))
                print("[WARN] 3D solver does not accept alpha volume; using mean(alpha) =", alpha_mean)
                u_dev, state, hist = admm_speckle_scaled_3d(
                    backend,
                    b=b_dev,
                    alpha=float(alpha_mean),
                    mu=float(cfg.mu),
                    beta=float(cfg.beta0),
                    n_admm_iters=int(cfg.admm),
                    n_pd_iters=int(cfg.pd),
                    pd_tol=float(cfg.pd_tol),
                    admm_tol=float(cfg.admm_tol),
                    auto_beta=bool(cfg.auto_beta),
                    verbose=True,
                )
            else:
                raise

        sec = time.time() - t0
        print(f"3D ADMM done in {sec:.2f} sec.")

        u01 = backend.to_cpu(u_dev).astype(np.float32)
        u01 = np.clip(u01, 0.0, 1.0)
        print("Denoised volume shape:", u01.shape)

        np.save(run_dir / "denoised_volume.npy", u01)
        print("Saved denoised volume:", (run_dir / "denoised_volume.npy").resolve())

        out_dicom = run_dir / f"{dicom_path.name}_denoised_full3d.dcm"
        saved = save_denoised_dicom(ds, u01, out_dicom, scale=scale)
        print("Saved denoised DICOM:", saved.resolve())

        cmp_dir = run_dir / "compare_slices"
        save_slice_comparisons(b01, u01, cmp_dir, slice_ids=slice_ids)
        print("Saved slice comparisons to:", cmp_dir.resolve())

        prev_dir = run_dir / "preview_png"
        prev_dir.mkdir(exist_ok=True)
        for z in slice_ids:
            Image.fromarray((b01[z] * 255).astype(np.uint8)).save(prev_dir / f"b_{z:04d}.png")
            Image.fromarray((u01[z] * 255).astype(np.uint8)).save(prev_dir / f"u_{z:04d}.png")
        print("Saved quick preview PNGs to:", prev_dir.resolve())


if __name__ == "__main__":
    main()
