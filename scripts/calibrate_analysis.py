"""Per-run analysis: turn a scripts/calibrate.py .npz into the summary
statistics described in the calibration spec (tau, tau_e, f_poison,
encounter_rate, g_measured, g_predicted, ratio). Used by
scripts/calibrate_all.py to build the sweep summary CSV; also runnable
standalone on a single .npz for debugging.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _agent_gaps(event_step, event_agent, n_agents):
    """Pooled inter-meal gaps across agents, excluding the interval before
    each agent's first recorded meal (np.diff on a sorted per-agent step
    list naturally drops that leading interval)."""
    gaps = []
    order = np.argsort(event_agent, kind="stable")
    event_agent = event_agent[order]
    event_step = event_step[order]
    for agent in range(n_agents):
        steps = np.sort(event_step[event_agent == agent])
        if len(steps) > 1:
            gaps.append(np.diff(steps))
    if not gaps:
        return np.array([], dtype=np.float64)
    return np.concatenate(gaps).astype(np.float64)


def bootstrap_ci_mean(data, n_boot=2000, ci=0.95, seed=0, max_resample_chunk=20_000_000):
    """Nonparametric bootstrap CI of the mean. Draws n_boot resamples of
    size n from `data`, done in chunks bounded to ~max_resample_chunk total
    index entries at a time -- a naive (n_boot, n) index array is fine for
    the lone-agent conditions (thousands of gaps) but is O(n_boot * n) and
    can reach billions of elements for the crowded conditions (n_agents=500
    /2000 produce hundreds of thousands to millions of meal-gap events),
    which is what actually happened here: it swelled to multiple GB and
    started swapping before being killed."""
    if len(data) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_boot)
    batch = max(1, max_resample_chunk // max(n, 1))
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        boot_means[start:start + b] = data[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def least_squares_slope_per_agent(energy):
    """energy: (steps, n_agents). Returns per-agent OLS slope of energy vs
    step index."""
    steps, n_agents = energy.shape
    t = np.arange(steps, dtype=np.float64)
    t_centered = t - t.mean()
    denom = (t_centered ** 2).sum()
    y_centered = energy - energy.mean(axis=0, keepdims=True)
    slopes = (t_centered[:, None] * y_centered).sum(axis=0) / denom
    return slopes


def g_predicted_fn(strategy, nutrition, energy_decay, poison_multiplier, tau, tau_e):
    if strategy == "wander":
        return -energy_decay
    elif strategy == "approach_all":
        if not np.isfinite(tau) or tau <= 0:
            return float("nan")
        return nutrition * (1 + poison_multiplier) / (2 * tau) - energy_decay
    elif strategy == "discriminate":
        if not np.isfinite(tau_e) or tau_e <= 0:
            return float("nan")
        return nutrition / tau_e - energy_decay
    raise ValueError(f"unknown strategy {strategy!r}")


def analyze_run(npz_path, n_boot=2000, boot_seed=0):
    d = np.load(npz_path)

    strategy = str(d["strategy"])
    n_agents = int(d["n_agents"])
    nb_mushrooms = int(d["nb_mushrooms"])
    poison_multiplier = float(d["poison_multiplier"])
    seed = int(d["seed"])
    nutrition = float(d["nutrition"])
    energy_decay = float(d["energy_decay"])
    steps = int(d["steps"])
    epsilon = float(d["epsilon"]) if "epsilon" in d.files else 0.0
    k = int(d["k"]) if "k" in d.files else None

    event_step = d["event_step"]
    event_agent = d["event_agent"]
    event_class = d["event_class"]  # 1 = edible, 2 = poison
    energy = d["energy"]

    gaps_all = _agent_gaps(event_step, event_agent, n_agents)
    edible_mask = event_class == 1
    gaps_e = _agent_gaps(event_step[edible_mask], event_agent[edible_mask], n_agents)

    tau_mean = float(gaps_all.mean()) if len(gaps_all) else float("nan")
    tau_median = float(np.median(gaps_all)) if len(gaps_all) else float("nan")
    tau_ci_lo, tau_ci_hi = bootstrap_ci_mean(gaps_all, n_boot=n_boot, seed=boot_seed)
    tau_e_mean = float(gaps_e.mean()) if len(gaps_e) else float("nan")

    n_meals = len(event_class)
    n_poison = int((event_class == 2).sum())
    f_poison = (n_poison / n_meals) if n_meals > 0 else float("nan")
    encounter_rate = n_meals / (n_agents * steps)

    slopes = least_squares_slope_per_agent(energy)
    g_measured = float(slopes.mean())

    g_predicted = g_predicted_fn(strategy, nutrition, energy_decay, poison_multiplier,
                                  tau_mean, tau_e_mean)

    if np.isfinite(g_predicted) and abs(g_predicted) >= 0.01:
        ratio = g_measured / g_predicted
    else:
        # g_predicted ~ 0 (the P = -1 boundary case): a ratio is meaningless
        # here, report the difference in its place.
        ratio = g_measured - g_predicted
        print(f"[{os.path.basename(npz_path)}] g_predicted ~= 0 "
              f"({g_predicted:.4f}); 'ratio' column holds g_measured - g_predicted instead")

    return dict(
        strategy=strategy, n_agents=n_agents, nb_mushrooms=nb_mushrooms,
        poison_multiplier=poison_multiplier, epsilon=epsilon, k=k, seed=seed,
        tau_mean=tau_mean, tau_median=tau_median, tau_ci_lo=tau_ci_lo, tau_ci_hi=tau_ci_hi,
        tau_e_mean=tau_e_mean, f_poison=f_poison, encounter_rate=encounter_rate,
        g_measured=g_measured, g_predicted=g_predicted, ratio=ratio,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz_path")
    args = p.parse_args()
    row = analyze_run(args.npz_path)
    for k, v in row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
