"""
Distribution of individual discrimination competency across a trained
population's final generation -- answers "do most agents sit around 0.4
disc, or is it more like 40% at ~1.0 and the rest near 0?"

Loads a saved final-population snapshot (*_agents.eqx / *_alive.npy, written
by sweep.py / ablation_sweep.py for any run that didn't go extinct) and
re-probes it at a higher replicate count than the in-sweep default, since
sweep.py only ever saves the population MEAN of the per-agent probe scores --
the per-agent array is computed internally by probe_population() but thrown
away before it hits disk.

Edit the constants below to point at the run you want, then run this file.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob

import jax
import numpy as np
import matplotlib.pyplot as plt

from archive.probe import probe_population, load_networks

# ---------------------------------------------------------------------------
# Which saved run to load
# ---------------------------------------------------------------------------
AGENTS_DIR = "/content/drive/MyDrive/sweep_recurrent_agents"  # directory holding *_agents.eqx / *_alive.npy
CONFIG_ID = 9
SEED = 2
RECURRENT = True   # must match how this population was actually trained
H_SIZE = 5

PROBE_REPLICATES = 32  # higher than the in-sweep default (8) for a cleaner per-agent estimate
PROBE_SEED = 12345

OUT_PATH = f"cfg{CONFIG_ID:03d}_seed{SEED}_competency_breakdown.png"

BLUE = "#2a78d6"
RED = "#e34948"
GRID = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"


def find_agent_files(agents_dir, config_id, seed):
    pattern = os.path.join(agents_dir, f"cfg{config_id:03d}_*_seed{seed}_agents.eqx")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern} -- check AGENTS_DIR/CONFIG_ID/SEED, and that "
            f"this run didn't go extinct (extinct runs don't save a final snapshot)."
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for cfg{config_id:03d} seed{seed} in {agents_dir}: {matches}")
    agents_path = matches[0]
    alive_path = agents_path.replace("_agents.eqx", "_alive.npy")
    return agents_path, alive_path


def summarize(name, values):
    v = np.asarray(values, dtype=float)
    print(f"\n{name}: n={len(v)}  mean={v.mean():.4f}  median={np.median(v):.4f}  std={v.std():.4f}")
    for thresh in (0.8, 0.5, 0.2):
        print(f"  fraction >= {thresh:>4}: {np.mean(v >= thresh):.1%}")
    print(f"  fraction in [-0.1, 0.1] (near-indifferent): {np.mean((v >= -0.1) & (v <= 0.1)):.1%}")
    print(f"  fraction <= -0.2 (poison-preferring): {np.mean(v <= -0.2):.1%}")


def main():
    agents_path, alive_path = find_agent_files(AGENTS_DIR, CONFIG_ID, SEED)
    print(f"Loading {agents_path}")
    networks, alive = load_networks(agents_path, alive_path, h_size=H_SIZE, recurrent=RECURRENT)

    key = jax.random.key(PROBE_SEED)
    result = probe_population(networks, key, n_replicates=PROBE_REPLICATES)

    eat_disc = result["eat_disc"][alive]
    approach_disc = result["approach_disc"][alive]

    summarize("eat_disc (individual eating discrimination)", eat_disc)
    summarize("approach_disc (individual approach discrimination)", approach_disc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#fcfcfb")

    bins = np.linspace(-1, 1, 41)

    ax = axes[0]
    ax.hist(eat_disc, bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(0, color=GRID, linewidth=1, linestyle="--")
    ax.axvline(eat_disc.mean(), color=INK, linewidth=1.5, label=f"mean = {eat_disc.mean():.3f}")
    ax.set_xlabel("eat_disc (individual)")
    ax.set_ylabel("number of agents")
    ax.set_title(f"cfg{CONFIG_ID} seed{SEED}: eating discrimination\n(n={alive.sum()} living agents)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(approach_disc, bins=bins, color=RED, edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(0, color=GRID, linewidth=1, linestyle="--")
    ax.axvline(approach_disc.mean(), color=INK, linewidth=1.5, label=f"mean = {approach_disc.mean():.3f}")
    ax.set_xlabel("approach_disc (individual)")
    ax.set_title("approach discrimination", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(approach_disc, eat_disc, s=10, color=BLUE, alpha=0.35, edgecolors="none")
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_xlabel("approach_disc")
    ax.set_ylabel("eat_disc")
    ax.set_title("per-agent: approach vs. eat discrimination", fontsize=10)

    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
