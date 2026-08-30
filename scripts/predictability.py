#!/usr/bin/env python3
"""Per-frame latency distribution for DOD vs OpenVINS.

The paper's claim is bounded, predictable per-frame state, but it reports mean
throughput -- which is exactly the statistic that hides a tail. Every run
already writes per-frame timings, so the distribution costs nothing to compute.

DOD writes  timestamp,tracking_ms,estimator_ms,total_ms,observations   (ms)
OpenVINS writes  # timestamp,tracking,propagation,...,total            (seconds)

Usage:  scripts/predictability.py <runs_dir> [--csv out.csv] [--fig out.png]
"""
import sys, csv, os, argparse
import numpy as np

# DOD stem -> OpenVINS stem, per sequence.
PAIRS = [
    ("MH_01", "MH_01_easy",      "ov_mh01"),
    ("MH_02", "MH_02_easy",      "ov_MH_02_easy"),
    ("MH_03", "MH_03_medium",    "ov_MH_03_medium"),
    ("MH_04", "MH_04_difficult", "ov_MH_04_difficult"),
    ("MH_05", "MH_05_difficult", "ov_MH_05_difficult"),
    ("V1_01", "V1_01_easy",      "ov_V1_01_easy"),
    ("V1_02", "V1_02_medium",    "ov_V1_02_medium"),
    ("V1_03", "V1_03_difficult", "ov_V1_03_difficult"),
    ("circle",   "KAIST_circle",   "ov_circle"),
    ("infinity", "KAIST_infinity", "ov_infinity"),
]


def load_dod(path):
    """total_ms column, already in milliseconds."""
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out.append(float(row["total_ms"]))
            except (KeyError, ValueError):
                pass
    return np.asarray(out)


def load_ov(path):
    """Last column is the per-frame total, in SECONDS -> convert to ms."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                out.append(float(parts[-1]) * 1000.0)
            except ValueError:
                pass
    return np.asarray(out)


def stats(a):
    return dict(n=len(a), mean=a.mean(), p50=np.percentile(a, 50),
                p95=np.percentile(a, 95), p99=np.percentile(a, 99),
                mx=a.max(), jitter=np.percentile(a, 99) - np.percentile(a, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    d = args.runs_dir

    rows, series = [], {}
    for name, dod_stem, ov_stem in PAIRS:
        dp = os.path.join(d, dod_stem + "_timing.csv")
        op = os.path.join(d, ov_stem + "_timing.txt")
        if not (os.path.exists(dp) and os.path.exists(op)):
            print(f"skip {name}: missing {'DOD' if not os.path.exists(dp) else 'OV'} timing")
            continue
        dod, ov = load_dod(dp), load_ov(op)
        if len(dod) == 0 or len(ov) == 0:
            print(f"skip {name}: empty")
            continue
        sd, so = stats(dod), stats(ov)
        series[name] = (dod, ov)
        rows.append((name, sd, so))

    hdr = (f"{'seq':<9} {'DOD p50':>8} {'p99':>8} {'max':>9} {'jit':>7} | "
           f"{'OV p50':>8} {'p99':>8} {'max':>9} {'jit':>7} | "
           f"{'p50x':>5} {'p99x':>5}")
    print("\nPer-frame latency, milliseconds\n" + hdr)
    print("-" * len(hdr))
    for name, sd, so in rows:
        print(f"{name:<9} {sd['p50']:8.3f} {sd['p99']:8.3f} {sd['mx']:9.3f} {sd['jitter']:7.3f} | "
              f"{so['p50']:8.3f} {so['p99']:8.3f} {so['mx']:9.3f} {so['jitter']:7.3f} | "
              f"{so['p50']/sd['p50']:5.2f} {so['p99']/sd['p99']:5.2f}")

    if rows:
        gp50 = np.exp(np.mean([np.log(so['p50']/sd['p50']) for _, sd, so in rows]))
        gp99 = np.exp(np.mean([np.log(so['p99']/sd['p99']) for _, sd, so in rows]))
        gjit = np.exp(np.mean([np.log(so['jitter']/sd['jitter']) for _, sd, so in rows]))
        print("-" * len(hdr))
        print(f"geometric mean speedup   p50 {gp50:.2f}x   p99 {gp99:.2f}x   jitter {gjit:.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "estimator", "n", "mean_ms", "p50_ms",
                        "p95_ms", "p99_ms", "max_ms", "jitter_ms"])
            for name, sd, so in rows:
                for est, s in (("DOD", sd), ("OpenVINS", so)):
                    w.writerow([name, est, s['n'], f"{s['mean']:.4f}", f"{s['p50']:.4f}",
                                f"{s['p95']:.4f}", f"{s['p99']:.4f}", f"{s['mx']:.4f}",
                                f"{s['jitter']:.4f}"])
        print(f"\nwrote {args.csv}")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(series)
        ncol = 5
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow), sharey=True)
        for ax, (name, (dod, ov)) in zip(np.ravel(axes), series.items()):
            for a, lab, c in ((dod, "DOD", "C0"), (ov, "OpenVINS", "C1")):
                xs = np.sort(a)
                ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab, color=c, lw=1.2)
            ax.set_xscale("log")
            # Explicit decade-ish ticks: matplotlib's default log minor labels
            # collide into an unreadable smear at this panel width.
            ax.set_xticks([1, 2, 5, 10, 20, 40])
            ax.set_xticklabels(["1", "2", "5", "10", "20", "40"], fontsize=8)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlim(1, 45)
            ax.set_title(name, fontsize=9)
            ax.grid(alpha=.3)
            ax.axhline(0.99, color="k", ls=":", lw=.7)
        for ax in np.ravel(axes)[n:]:
            ax.axis("off")
        np.ravel(axes)[0].set_ylabel("CDF")
        for ax in np.ravel(axes)[max(0, n - ncol):n]:
            ax.set_xlabel("per-frame latency (ms)", fontsize=8)
        np.ravel(axes)[0].legend(fontsize=7)
        fig.suptitle("Per-frame latency CDF (log ms); dotted line = p99", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=150)
        print(f"wrote {args.fig}")


if __name__ == "__main__":
    main()
