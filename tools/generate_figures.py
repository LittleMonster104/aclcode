#!/usr/bin/env python3
"""Generate convergence figures from resnet_ablation_results.json."""
import json, os, numpy as np

BASE = "/workspace"
RES_PTH = os.path.join(BASE, "experiments/cifar_ablation_quick/resnet_ablation_results.json")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(RES_PTH) as f:
    data = json.load(f)

results = data["results"]
ranks = sorted(results.keys(), key=int)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: InfoNCE convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in ranks:
        hist = results[r]["history"]
        ax.plot(hist["info_nce"], label=f"Rank {r}", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("InfoNCE Convergence Across Ranks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "infonce_convergence.png"), dpi=150)
    print("Saved infonce_convergence.png")

    # Figure 2: DistPres convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in ranks:
        hist = results[r]["history"]
        ax.plot(hist["dist_pres"], label=f"Rank {r}", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DistPres Loss")
    ax.set_title("Distance Preservation Convergence Across Ranks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "distpres_convergence.png"), dpi=150)
    print("Saved distpres_convergence.png")

    # Figure 3: Recall@K bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ranks))
    width = 0.25
    r1s, r5s, r10s = [], [], []
    for i, r in enumerate(ranks):
        r1s.append(results[r]["recall_at_1"] * 100)
        r5s.append(results[r]["recall_at_5"] * 100)
        r10s.append(results[r]["recall_at_10"] * 100)
    ax.bar(x - width, r1s, width, label="R@1", color="#4C72B0")
    ax.bar(x, r5s, width, label="R@5", color="#DD8452")
    ax.bar(x + width, r10s, width, label="R@10", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {r}" for r in ranks])
    ax.set_ylabel("Recall (%)")
    ax.set_title("Retrieval Performance by Rank")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "recall_by_rank.png"), dpi=150)
    print("Saved recall_by_rank.png")

except ImportError:
    print("matplotlib not available; printing summary table instead:")
    print(f"{'Rank':>6} {'R@1 (%)':>8} {'R@5 (%)':>8} {'R@10 (%)':>9} {'InfoNCE':>10}")
    for r in ranks:
        d = results[r]
        print(f"{int(r):>6} {d['recall_at_1']*100:>8.1f} {d['recall_at_5']*100:>8.1f} "
              f"{d['recall_at_10']*100:>9.1f} {d['info_nce_final']:>10.6f}")
