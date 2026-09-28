"""
Experiment 1 (perception vs. motor bottleneck): does a single forward pass
through the hidden layer/state linearly separate edible vs. poisonous
mushroom variants, for a saved trained population?

Builds synthetic input vectors directly -- no grid simulation, no rollout --
for all 20 single-bit-flip mushroom variants (10 poison: feat_idx 0-9, 10
edible: feat_idx 10-19; MUSH_LIBRARY has no un-flipped prototype row, see
mushroom_world.py's bit_change = jnp.eye(10)) x 8 bearing angles = 160 rows.
Observation layout is [cos, sin, features(10), signal(3, neutral), pain/
pleasure(2, zeroed)] per mushroom_world.py:219.

For the recurrent architecture this is t=1 from h_0=0. Recurrent_Network.h2h
has use_bias=False, so h2h(0) = 0 exactly, meaning Z_rec at t=1 reduces to
tanh(i2h(x)) -- structurally the same shape as feedforward's sigmoid(i2h(x))
(one Linear + one pointwise nonlinearity), so this is a fair, dynamics-free
comparison: recurrence hasn't had a chance to do anything temporal yet, only
representational capacity is being tested at this stage.

For a subsample of living agents in each saved population, fits a
StratifiedGroupKFold(5) + L2-logistic-regression decoder per representation:
Z_raw (the raw 10-bit feature vector, a ceiling reference -- the two
prototypes are exact bitwise complements, so this is trivially near-linearly-
separable and any representation scoring well below it is more entangled
than the raw input), Z_ff, and Z_rec. Grouped so all 8 headings of a variant
stay in the same fold, otherwise the classifier could exploit heading-
specific artifacts instead of decoding the class itself.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob

import numpy as np
import pandas as pd
import jax.numpy as jnp
import equinox as eqx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import matplotlib.pyplot as plt

from archive.probe import load_networks, NEUTRAL_SIGNAL, NEUTRAL_OUTCOME
from mushroom_world import MUSH_LIBRARY

# ---------------------------------------------------------------------------
FF_AGENTS_DIR = "/content/drive/MyDrive/sweep_agents"
REC_AGENTS_DIR = "/content/drive/MyDrive/sweep_recurrent_agents"
CONFIG_ID = 9
SEED = 2
H_SIZE = 5

N_HEADINGS = 8
N_AGENTS_TO_PROBE = 50
N_FOLDS = 5
RANDOM_SEED = 0

OUT_PREFIX = f"cfg{CONFIG_ID:03d}_seed{SEED}"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRID = "#c3c2b7"


def build_dataset():
    angles = np.linspace(0, 2 * np.pi, N_HEADINGS, endpoint=False)
    cos_vals, sin_vals = np.cos(angles), np.sin(angles)

    feat_indices = list(range(20))  # 0-9 poison, 10-19 edible
    labels_per_variant = np.array([0] * 10 + [1] * 10)  # 0=poison, 1=edible

    signal = np.asarray(NEUTRAL_SIGNAL)
    outcome = np.asarray(NEUTRAL_OUTCOME)

    rows, y, groups = [], [], []
    for group_id, feat_idx in enumerate(feat_indices):
        feat_vec = np.asarray(MUSH_LIBRARY[feat_idx])
        for h in range(N_HEADINGS):
            obs = np.concatenate([[cos_vals[h]], [sin_vals[h]], feat_vec, signal, outcome])
            rows.append(obs)
            y.append(labels_per_variant[group_id])
            groups.append(group_id)

    X_obs = np.stack(rows).astype(np.float32)  # (160, 17)
    y = np.array(y)
    groups = np.array(groups)
    Z_raw = X_obs[:, 2:12]  # the raw 10-bit feature slice
    return X_obs, Z_raw, y, groups


def compute_hidden_reps(networks, X_obs, recurrent):
    X_jax = jnp.asarray(X_obs)

    if recurrent:
        def per_agent(net):
            def one(x):
                _, hidden = net(x, net.init_hidden())
                return hidden
            return eqx.filter_vmap(one)(X_jax)
    else:
        def per_agent(net):
            def one(x):
                h = net.layers[0](x)
                h = net.layers[1](h)
                return h
            return eqx.filter_vmap(one)(X_jax)

    Z = eqx.filter_vmap(per_agent)(networks)  # (MAX_AGENTS, 160, H_SIZE)
    return np.asarray(Z)


def decode_score(Z, y, groups, n_folds=N_FOLDS):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)  # penalty defaults to l2
    acc = cross_val_score(clf, Z, y, groups=groups, cv=sgkf, scoring="accuracy")
    auc = cross_val_score(clf, Z, y, groups=groups, cv=sgkf, scoring="roc_auc")
    return acc.mean(), auc.mean()


def load_population(agents_dir, config_id, seed, recurrent):
    pattern = f"{agents_dir}/cfg{config_id:03d}_*_seed{seed}_agents.eqx"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern} -- check the path, and that this run "
            f"didn't go extinct (extinct runs don't save a final snapshot)."
        )
    agents_path = matches[0]
    alive_path = agents_path.replace("_agents.eqx", "_alive.npy")
    print(f"Loading {agents_path}")
    return load_networks(agents_path, alive_path, h_size=H_SIZE, recurrent=recurrent)


def probe_architecture(agents_dir, recurrent, X_obs, y, groups, rng):
    networks, alive = load_population(agents_dir, CONFIG_ID, SEED, recurrent)
    alive_idx = np.flatnonzero(alive)
    n_probe = min(N_AGENTS_TO_PROBE, len(alive_idx))
    probe_idx = rng.choice(alive_idx, size=n_probe, replace=False)
    print(f"  probing {n_probe} of {len(alive_idx)} living agents "
          f"({'recurrent' if recurrent else 'feedforward'})")

    Z_hidden = compute_hidden_reps(networks, X_obs, recurrent)

    rows = []
    for idx in probe_idx:
        acc, auc = decode_score(Z_hidden[idx], y, groups)
        rows.append({"agent_idx": int(idx), "accuracy": acc, "auc": auc})
    return pd.DataFrame(rows)


def main():
    X_obs, Z_raw, y, groups = build_dataset()
    print(f"Dataset: {X_obs.shape[0]} rows ({len(set(groups))} variants x {N_HEADINGS} headings)")

    raw_acc, raw_auc = decode_score(Z_raw, y, groups)
    print(f"\nZ_raw (10-bit feature, ceiling reference): accuracy={raw_acc:.3f}  AUC={raw_auc:.3f}")

    rng = np.random.default_rng(RANDOM_SEED)

    print("\nFeedforward:")
    ff_results = probe_architecture(FF_AGENTS_DIR, False, X_obs, y, groups, rng)
    print(f"  Z_ff: mean accuracy={ff_results.accuracy.mean():.3f} (+/-{ff_results.accuracy.std():.3f}), "
          f"mean AUC={ff_results.auc.mean():.3f} (+/-{ff_results.auc.std():.3f})")

    print("\nRecurrent:")
    rec_results = probe_architecture(REC_AGENTS_DIR, True, X_obs, y, groups, rng)
    print(f"  Z_rec: mean accuracy={rec_results.accuracy.mean():.3f} (+/-{rec_results.accuracy.std():.3f}), "
          f"mean AUC={rec_results.auc.mean():.3f} (+/-{rec_results.auc.std():.3f})")

    ff_results["representation"] = "Z_ff"
    rec_results["representation"] = "Z_rec"
    combined = pd.concat([ff_results, rec_results], ignore_index=True)
    out_csv = f"{OUT_PREFIX}_linear_decode.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.axhline(raw_acc, color=MUTED, linewidth=1.5, linestyle="--", label=f"Z_raw ceiling ({raw_acc:.2f})")
    ax.axhline(0.5, color=GRID, linewidth=1, linestyle=":", label="chance")

    for pos, (label, df, color) in enumerate([
        ("Z_ff", ff_results, ORANGE),
        ("Z_rec", rec_results, BLUE),
    ]):
        parts = ax.violinplot([df.accuracy.values], positions=[pos], showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        for key in ("cmeans", "cmaxes", "cmins", "cbars"):
            parts[key].set_color(color)
        jitter = np.random.default_rng(1).normal(0, 0.03, size=len(df))
        ax.scatter(pos + jitter, df.accuracy.values, s=14, color=color, alpha=0.5, edgecolors="none")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["feedforward\n(Z_ff)", "recurrent\n(Z_rec)"])
    ax.set_ylabel("cross-validated decoding accuracy")
    ax.set_title(f"cfg{CONFIG_ID} seed{SEED}: linear decodability of edible vs. poison\n"
                 f"(t=1, h0=0, n={N_AGENTS_TO_PROBE} agents per architecture)", fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    plt.tight_layout()
    out_png = f"{OUT_PREFIX}_linear_decode.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
