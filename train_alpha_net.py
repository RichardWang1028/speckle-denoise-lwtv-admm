# train_alpha_net.py
from __future__ import annotations

import argparse
from alpha_net import train_alpha_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patch", type=int, default=256)

    ap.add_argument("--alpha_min", type=float, default=0.2)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--k", type=float, default=25.0)

    ap.add_argument("--nppi", type=int, default=16, help="patches per image")
    ap.add_argument("--var_min", type=float, default=0.005)
    ap.add_argument("--var_max", type=float, default=0.03)

    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")

    args = ap.parse_args()

    train_alpha_net(
        clean_dir=args.clean_dir,
        out_ckpt=args.ckpt,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        patch=args.patch,
        n_patches_per_image=args.nppi,
        var_range=(args.var_min, args.var_max),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        k=args.k,
        num_workers=args.workers,
        amp=bool(args.amp),
    )


if __name__ == "__main__":
    main()
