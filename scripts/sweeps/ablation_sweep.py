"""
Targeted 2x2 ablation: recurrence x pain/pleasure input, on a curated subset
of configs, to disentangle the two effects that got bundled together in the
full "recurrent" sweep (which had PAIN_PLEASURE=True baked in alongside
RECURRENT=True), and to get the feedforward baseline back onto the current
code path (the original feedforward sweep used a real 15-dim input; the
current code always uses input_dim=17 with the pain/pleasure channels
zeroed out when pain_pleasure=False -- functionally near-identical but not
the same code path/PRNG stream, so it isn't safe to just reuse the old CSV).

The four cells of the design are:
    ff_nopp   recurrent=False, pain_pleasure=False   <- run here (fresh, current code)
    ff_pp     recurrent=False, pain_pleasure=True     <- run here
    rec_nopp  recurrent=True,  pain_pleasure=False    <- run here
    rec_pp    recurrent=True,  pain_pleasure=True     (already have: recurrent.tsv,
                                                         generated under the current
                                                         17-input code -- backfilled
                                                         at summarize time, not rerun)

Config generation is a verbatim copy of sweep.py's generate_configs (same
LHS_SEED=42, N_CONFIGS=100) so parameter values for a given config_id line
up exactly with the existing feedforward/recurrent sweep data. NOT imported
from sweep.py directly, since sweep.py runs its full sweep as a top-level
side effect on import.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import hashlib

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.stats import qmc

from mushroom_world import MushroomWorld
from archive.probe import probe_population

# ---------------------------------------------------------------------------
# Fixed params -- must match sweep.py's fixed params for comparability
# ---------------------------------------------------------------------------
SX = 100
SY = 100
NB_AGENTS = 2000
MAX_AGENTS = 5000
NB_MUSHROOMS = 100
ENERGY_START = 200.0
MUSHROOM_NUTRITION = 20.0
REPROD_THRESHOLD = 210.
PERC_RADIUS = 10
SHUFFLE_PERIOD = 200
NO_SIGNAL = True
H_SIZE = 5
STEPS = 100_000
CHUNK = 100
PROBE_EVERY = 10
PROBE_REPLICATES = 8

N_CONFIGS = 100
N_SEEDS = 5
LHS_SEED = 42

PROBE_KEYS = (
    "edible_approach", "edible_eat_rate", "poison_approach",
    "poison_eat_rate", "approach_disc", "eat_disc",
)

# ---------------------------------------------------------------------------
# Curated config selection (n=20) -- edit this list to change what gets run.
# Built from the paired feedforward vs. recurrent+pain/pleasure comparison:
#   - top 10 configs by gain in individual (probe eat) discrimination
#   - bottom 7 configs (recurrent+pp did worse than feedforward -- negative
#     controls, important for not overclaiming a universal effect)
#   - config 97, the abstract's highlighted decoupled case
#   - 2 fillers spread across energy_decay quartiles, to keep coverage of
#     the "permissive region" story rather than only cherry-picked extremes
# ---------------------------------------------------------------------------
SELECTED_CONFIG_IDS = sorted(set([
    3, 5, 6, 8, 9, 15, 19, 24, 32, 49, 52, 60, 62, 69, 77, 81, 83, 95, 97, 99,
]))

# Which of the 4 cells to actually simulate.
CONDITIONS_TO_RUN = [
    # (tag,       recurrent, pain_pleasure)
    ("ff_nopp",   False,     False),
    ("ff_pp",     False,     True),
    ("rec_nopp",  True,      False),
    # ("rec_pp",  True,      True),  # already have this: recurrent.tsv (current 17-input code)
]

# Path to your existing recurrent-sweep summary CSV, used to backfill the
# rec_pp rows for the selected configs at summarize time. Set to None to
# skip backfilling (summary will then only contain the 3 conditions run
# above).
EXISTING_SUMMARY_PATHS = {
    "rec_pp": None,  # e.g. "/content/drive/MyDrive/sweep_recurrent/config_summary_probed.csv"
}

RESULTS_DIR = "/content/drive/MyDrive/sweep_ablation"
AGENTS_DIR = "/content/drive/MyDrive/sweep_ablation_agents"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AGENTS_DIR, exist_ok=True)


def generate_configs(n_configs, seed):
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_configs)

    configs = []
    for i, s in enumerate(samples):
        mutation_std = float(10 ** (np.log10(0.005) + s[0] * (np.log10(0.2) - np.log10(0.005))))
        poison_multiplier = float(-2 - s[1] * 6)
        reprod_cost = float(s[2] * 12)
        energy_decay = float(0.05 + s[3] * 0.10)

        configs.append({
            "config_id": i,
            "mutation_std": mutation_std,
            "poison_multiplier": poison_multiplier,
            "reprod_cost": reprod_cost,
            "energy_decay": energy_decay,
        })
    return configs


def config_hash(cfg):
    s = f"{cfg['mutation_std']:.6f}_{cfg['poison_multiplier']:.4f}_{cfg['reprod_cost']:.4f}_{cfg['energy_decay']:.4f}"
    return hashlib.md5(s.encode()).hexdigest()[:8]


def run_one(cfg, seed, recurrent, pain_pleasure):
    key = jax.random.key(seed)
    probe_key = jax.random.key(seed + 1_000_003)
    env = MushroomWorld(
        seed=seed, grid_x=SX, grid_y=SY,
        nb_agents=NB_AGENTS, max_agents=MAX_AGENTS, nb_mushrooms=NB_MUSHROOMS,
        energy_start=ENERGY_START, energy_decay=cfg["energy_decay"],
        mushroom_nutrition=MUSHROOM_NUTRITION,
        reprod_threshold=REPROD_THRESHOLD, reprod_cost=cfg["reprod_cost"],
        shuffle_period=SHUFFLE_PERIOD,
        mutation_std=cfg["mutation_std"],
        poison_multiplier=cfg["poison_multiplier"],
        frozen_baseline=False,
        no_signal=NO_SIGNAL,
        recurrent=recurrent,
        pain_pleasure=pain_pleasure,
        h_size=H_SIZE,
    )
    agents, mushrooms = env.reset_fn()
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)

    def step(carry, _):
        key, dynamic_agents, mushrooms = carry
        agents = eqx.combine(dynamic_agents, static_agents)
        key, sk1, sk2, sk3, sk4 = jax.random.split(key, 5)

        obs = env._compute_obs(sk1, agents, mushrooms, PERC_RADIUS)
        probs, hidden = eqx.filter_vmap(lambda n, o, h: n(o, h))(agents.network, obs, agents.hidden)
        actions = jax.random.bernoulli(sk2, probs).astype(jnp.int32)
        agents = eqx.tree_at(lambda a: a.hidden, agents, hidden)

        agents, mushrooms, edible, poisonous = env._compute_update(sk3, actions, agents, mushrooms)
        agents = env._compute_reproduce(sk4, agents, env.mutation_std)

        n_alive = agents.alive.sum()
        mean_energy = jnp.where(n_alive > 0, (agents.energy * agents.alive).sum() / n_alive, 0.0)

        dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
        return (key, dynamic_agents, mushrooms), (n_alive, mean_energy, edible, poisonous)

    @eqx.filter_jit
    def run_chunk(key, dynamic_agents, mushrooms):
        return jax.lax.scan(step, (key, dynamic_agents, mushrooms), None, length=CHUNK)

    def run_probe(dynamic_agents, key):
        agents = eqx.combine(dynamic_agents, static_agents)
        alive_np = np.array(agents.alive).astype(bool)
        n_alive = int(alive_np.sum())

        if n_alive == 0:
            return n_alive, {k: float('nan') for k in PROBE_KEYS}

        probe_result = probe_population(agents.network, key, n_replicates=PROBE_REPLICATES)
        summary = {k: float(np.mean(np.asarray(v)[alive_np])) for k, v in probe_result.items()}
        return n_alive, summary

    chunk_edible, chunk_poisonous, chunk_disc, chunk_alive, chunk_energy = [], [], [], [], []
    probe_chunks, probe_n_alive = [], []
    probe_series = {k: [] for k in PROBE_KEYS}
    extinction_chunk = None

    n_chunks = STEPS // CHUNK

    for c in range(n_chunks):
        (key, dynamic_agents, mushrooms), (n_alive, mean_energy, edible, poisonous) = run_chunk(key, dynamic_agents, mushrooms)

        total_e = int(np.array(edible).sum())
        total_p = int(np.array(poisonous).sum())
        total = total_e + total_p
        disc = (total_e - total_p) / total if total > 0 else float('nan')

        chunk_edible.append(total_e)
        chunk_poisonous.append(total_p)
        chunk_disc.append(disc)
        chunk_alive.append(float(np.array(n_alive).mean()))
        chunk_energy.append(float(np.array(mean_energy).mean()))

        is_extinct = int(np.array(n_alive)[-1]) == 0
        is_last = (c == n_chunks - 1)

        if c % PROBE_EVERY == 0 or is_extinct or is_last:
            probe_key, subkey = jax.random.split(probe_key)
            n_alive_probe, probe_summary = run_probe(dynamic_agents, subkey)
            probe_chunks.append(c)
            probe_n_alive.append(n_alive_probe)
            for k, v in probe_summary.items():
                probe_series[k].append(v)

        if is_extinct:
            extinction_chunk = c
            print(f"  extinction at chunk {c}")
            break

    final_agents = eqx.combine(dynamic_agents, static_agents)

    result = {
        "config": cfg,
        "seed": seed,
        "recurrent": recurrent,
        "pain_pleasure": pain_pleasure,
        "chunk_edible": chunk_edible,
        "chunk_poisonous": chunk_poisonous,
        "chunk_disc": chunk_disc,
        "chunk_alive": chunk_alive,
        "chunk_energy": chunk_energy,
        "extinction_chunk": extinction_chunk,
        "completed_chunks": len(chunk_disc),
        "probe_chunks": probe_chunks,
        "probe_n_alive": probe_n_alive,
        **{f"probe_{k}": v for k, v in probe_series.items()},
    }
    return result, final_agents


def run_ablation():
    all_configs = {c["config_id"]: c for c in generate_configs(N_CONFIGS, LHS_SEED)}
    selected = [all_configs[i] for i in SELECTED_CONFIG_IDS]

    with open(os.path.join(RESULTS_DIR, "configs.json"), "w") as f:
        json.dump(selected, f, indent=2)

    total_runs = len(selected) * len(CONDITIONS_TO_RUN) * N_SEEDS
    done = 0

    for tag, recurrent, pain_pleasure in CONDITIONS_TO_RUN:
        for cfg in selected:
            cfg_hash = config_hash(cfg)
            for seed in range(N_SEEDS):
                out_path = os.path.join(RESULTS_DIR, f"cfg{cfg['config_id']:03d}_{tag}_{cfg_hash}_seed{seed}.json")
                if os.path.exists(out_path):
                    done += 1
                    continue

                print(f"[{done+1}/{total_runs}] {tag} cfg {cfg['config_id']} seed {seed}: "
                      f"mut={cfg['mutation_std']:.4f} poison={cfg['poison_multiplier']:.2f} "
                      f"reprod={cfg['reprod_cost']:.1f} decay={cfg['energy_decay']:.3f}")

                result, final_agents = run_one(cfg, seed, recurrent, pain_pleasure)

                final_window = [d for d in result["chunk_disc"][int(len(result["chunk_disc"]) * 0.8):] if not np.isnan(d)]
                final_disc = np.mean(final_window) if final_window else float('nan')
                final_probe_disc = result["probe_approach_disc"][-1] if result["probe_approach_disc"] else float('nan')
                print(f"  -> final env discrimination: {final_disc:.4f}  final probe discrimination: {final_probe_disc:.4f}")

                if result["extinction_chunk"] is None:
                    base = f"cfg{cfg['config_id']:03d}_{tag}_{cfg_hash}_seed{seed}"
                    eqx.tree_serialise_leaves(os.path.join(AGENTS_DIR, f"{base}_agents.eqx"), final_agents.network)
                    np.save(os.path.join(AGENTS_DIR, f"{base}_alive.npy"), np.array(final_agents.alive))

                with open(out_path, "w") as f:
                    json.dump(result, f)
                done += 1

    print(f"\nAblation runs complete. {total_runs} runs saved to {RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Summary aggregation -- same definitions as sweep_plot.py's final_stat /
# last_valid / best_seed, so numbers are directly comparable to the existing
# feedforward.tsv / recurrent.tsv tables.
# ---------------------------------------------------------------------------
def final_stat(values, frac=0.2):
    vals = [v for v in values[int(len(values) * (1 - frac)):] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def last_valid(values):
    vals = [v for v in values if not np.isnan(v)]
    return vals[-1] if vals else float("nan")


def best_seed(seed_ids, values):
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    if not valid.any():
        return None, float("nan")
    idx = np.flatnonzero(valid)[np.argmax(values[valid])]
    return seed_ids[idx], float(values[idx])


def argmax_valid(xs, ys):
    ys = np.asarray(ys, dtype=float)
    valid = ~np.isnan(ys)
    if not valid.any():
        return float("nan"), float("nan")
    idx = np.flatnonzero(valid)[np.nanargmax(ys[valid])]
    return xs[idx], float(ys[idx])


def summarize_condition(results_dir, tag, configs_by_id):
    """Aggregate all cfg*_{tag}_*.json files in results_dir into summary rows."""
    runs = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith("cfg") or not fname.endswith(".json") or fname == "configs.json":
            continue
        if f"_{tag}_" not in fname:
            continue
        cfg_id = int(fname[3:6])
        seed = int(fname.split("_seed")[1].split(".")[0])
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        runs.setdefault(cfg_id, {})[seed] = data

    rows = []
    for cfg_id, seeds in sorted(runs.items()):
        cfg = configs_by_id[cfg_id]
        seed_ids, seed_final_env, seed_final_approach, seed_final_eat = [], [], [], []
        seed_max_env, seed_max_approach, seed_max_eat, extinct_flags = [], [], [], []

        for seed, data in sorted(seeds.items()):
            seed_ids.append(seed)
            chunks = np.arange(len(data["chunk_alive"]))
            seed_final_env.append(final_stat(data["chunk_disc"]))
            seed_final_approach.append(last_valid(data["probe_approach_disc"]))
            seed_final_eat.append(last_valid(data["probe_eat_disc"]))

            _, env_peak_y = argmax_valid(chunks, data["chunk_disc"])
            _, approach_peak_y = argmax_valid(data["probe_chunks"], data["probe_approach_disc"])
            _, eat_peak_y = argmax_valid(data["probe_chunks"], data["probe_eat_disc"])
            seed_max_env.append(env_peak_y)
            seed_max_approach.append(approach_peak_y)
            seed_max_eat.append(eat_peak_y)

            extinct_flags.append(data["extinction_chunk"] is not None)

        best_env_seed, best_env_val = best_seed(seed_ids, seed_final_env)
        best_approach_seed, best_approach_val = best_seed(seed_ids, seed_final_approach)
        best_eat_seed, best_eat_val = best_seed(seed_ids, seed_final_eat)

        rows.append({
            "config_id": cfg_id,
            **{k: cfg[k] for k in ("mutation_std", "poison_multiplier", "reprod_cost", "energy_decay")},
            "n_seeds": len(seeds),
            "extinction_rate": float(np.mean(extinct_flags)),
            "mean_final_env_disc": float(np.nanmean(seed_final_env)),
            "mean_final_probe_approach_disc": float(np.nanmean(seed_final_approach)),
            "mean_final_probe_eat_disc": float(np.nanmean(seed_final_eat)),
            "mean_max_env_disc": float(np.nanmean(seed_max_env)),
            "mean_max_probe_approach_disc": float(np.nanmean(seed_max_approach)),
            "mean_max_probe_eat_disc": float(np.nanmean(seed_max_eat)),
            "best_seed_final_env_disc": best_env_val,
            "best_seed_final_env_disc_seed": best_env_seed,
            "best_seed_final_probe_approach_disc": best_approach_val,
            "best_seed_final_probe_approach_disc_seed": best_approach_seed,
            "best_seed_final_probe_eat_disc": best_eat_val,
            "best_seed_final_probe_eat_disc_seed": best_eat_seed,
        })

    return rows


def summarize_ablation():
    all_configs = {c["config_id"]: c for c in generate_configs(N_CONFIGS, LHS_SEED)}
    condition_meta = {
        "ff_nopp":  dict(recurrent=False, pain_pleasure=False),
        "ff_pp":    dict(recurrent=False, pain_pleasure=True),
        "rec_nopp": dict(recurrent=True,  pain_pleasure=False),
        "rec_pp":   dict(recurrent=True,  pain_pleasure=True),
    }

    all_rows = []

    # conditions actually simulated by this script
    for tag, recurrent, pain_pleasure in CONDITIONS_TO_RUN:
        for row in summarize_condition(RESULTS_DIR, tag, all_configs):
            row["condition"] = tag
            row["recurrent"] = recurrent
            row["pain_pleasure"] = pain_pleasure
            all_rows.append(row)

    # backfill from existing full sweep(s) for any cell not run above,
    # restricted to the selected config_ids
    run_tags = [c[0] for c in CONDITIONS_TO_RUN]
    for tag, path in EXISTING_SUMMARY_PATHS.items():
        if path is None or tag in run_tags:
            continue
        existing = pd.read_csv(path)
        existing = existing[existing.config_id.isin(SELECTED_CONFIG_IDS)].copy()
        existing["condition"] = tag
        existing["recurrent"] = condition_meta[tag]["recurrent"]
        existing["pain_pleasure"] = condition_meta[tag]["pain_pleasure"]
        all_rows.extend(existing.to_dict("records"))

    summary = pd.DataFrame(all_rows).sort_values(["config_id", "condition"])
    out_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Wrote {len(summary)} rows ({summary.condition.nunique()} conditions x "
          f"{summary.config_id.nunique()} configs) to {out_path}")
    return summary


if __name__ == "__main__":
    run_ablation()
    summarize_ablation()
