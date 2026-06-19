"""
draw_pr_curves.py - Reproduce paper Figure 2: precision-recall curves
overlaying Coin with the open-source LLM baselines on the same axes.

Reads one or more sidecar pickles produced by the evaluators:
  - eval_repro.py            -> data.pkl                 (Coin)
  - eval_open_baseline.py    -> <output>_probs.pkl       (vanilla LLMs)

Each pickle stores (probs, labels). For the Coin classifier pickle,
probs is a list of pairs [p_safe, p_unsafe]; for the open-source
baseline pickle, probs is a flat list of P(unsafe) scalars. Both
shapes are auto-detected.

Usage:
    python draw_pr_curves.py \
        --inputs Coin:data.pkl \
                 'Llama 3.2 3B:/tmp/coin_baselines/open_llama32_probs.pkl' \
                 'Llama 3.1 8B:/tmp/coin_baselines/open_llama31_8b_probs.pkl' \
                 'Qwen3 4B:/tmp/coin_baselines/open_qwen3_4b_probs.pkl' \
        --output /tmp/coin_pr_curves.png

Output: a single PNG/PDF (recall on x, precision on y) with one curve
per model. Each legend entry includes the model name and its AUPRC.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

# Distinguishable colors matching paper Fig. 2 (Coin highlighted)
DEFAULT_COLORS = {
    "coin":   "#d62728",   # red — emphasized
    "llama":  "#1f77b4",   # blue
    "qwen":   "#2ca02c",   # green
    "other":  "#ff7f0e",   # orange
    "gpt":    "#9467bd",   # purple
    "claude": "#8c564b",   # brown
}


def auto_color(name: str) -> str:
    low = name.lower()
    for key, col in DEFAULT_COLORS.items():
        if key in low:
            return col
    return None


def load_probs(path: Path):
    """Return (p_unsafe: np.ndarray, labels: np.ndarray)."""
    with open(path, "rb") as f:
        probs, labels = pickle.load(f)
    arr = np.asarray(probs)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        p_unsafe = arr[:, 1].astype(float)
    else:
        p_unsafe = arr.astype(float).reshape(-1)
    return p_unsafe, np.asarray(labels).astype(int)


def merge_shards(paths):
    """If user passes a glob-like list of shard pickles, concatenate them."""
    all_p, all_l = [], []
    for p in paths:
        pu, lb = load_probs(Path(p))
        all_p.append(pu); all_l.append(lb)
    return np.concatenate(all_p), np.concatenate(all_l)


def parse_input(spec: str):
    """'Coin:data.pkl' or 'Coin:data_shard0.pkl,data_shard1.pkl,...'"""
    if ":" not in spec:
        sys.exit(f"--inputs entries need NAME:PATH form, got: {spec}")
    name, paths = spec.split(":", 1)
    return name.strip(), [p.strip() for p in paths.split(",") if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="one or more NAME:PATH[,PATH,...] entries")
    ap.add_argument("--output", default="/tmp/coin_pr_curves.png")
    ap.add_argument("--title", default="Precision-Recall on MUF detection")
    ap.add_argument("--xlim", nargs=2, type=float, default=[0.0, 1.0])
    ap.add_argument("--ylim", nargs=2, type=float, default=[0.0, 1.0])
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 5))
    rows = []

    for spec in args.inputs:
        name, paths = parse_input(spec)
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            print(f"WARNING: skipping '{name}' — missing: {missing}",
                  file=sys.stderr)
            continue
        try:
            p_unsafe, labels = merge_shards(paths)
        except Exception as e:
            print(f"WARNING: failed to load '{name}': {e}", file=sys.stderr)
            continue
        if labels.sum() == 0:
            print(f"WARNING: '{name}' has no positive labels — skipping",
                  file=sys.stderr)
            continue

        auprc = float(average_precision_score(labels, p_unsafe))
        prec, rec, _ = precision_recall_curve(labels, p_unsafe)
        color = auto_color(name)
        is_coin = "coin" in name.lower()
        ax.plot(
            rec, prec,
            label=f"{name} (AUPRC = {auprc:.3f})",
            color=color,
            linewidth=2.5 if is_coin else 1.5,
            linestyle="-" if is_coin else "--",
            zorder=3 if is_coin else 2,
        )
        rows.append((name, auprc, len(labels), int(labels.sum())))

    if not rows:
        sys.exit("No curves were plotted — check --inputs paths.")

    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = Path(args.output)
    fig.savefig(out, dpi=200)
    if out.suffix != ".pdf":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"\nSaved: {out}")
    if out.suffix != ".pdf":
        print(f"       {out.with_suffix('.pdf')}")
    print("\nSummary:")
    print(f"{'Model':<25s} {'AUPRC':>8s} {'N':>8s} {'#pos':>8s}")
    for name, auprc, n, npos in rows:
        print(f"{name:<25s} {auprc:>8.4f} {n:>8d} {npos:>8d}")


if __name__ == "__main__":
    main()
