import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mushroom_world import MushroomWorld
import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np
import json
import os
import hashlib
from scipy.stats import qmc

from archive.probe import probe_population

# Fixed params
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
NO_SIGNAL = True          # flip to False to sweep the with-signal condition
RECURRENT = True         # flip to True to sweep the recurrent-network condition
PAIN_PLEASURE = True      # flip to False to sweep without the pain/pleasure input
H_SIZE = 5                # hidden layer width -- fixed per sweep run, not part of the LHS
                          # (interacts with mutation_std and forces recompilation if varied per-config)
STEPS = 100_000
CHUNK = 100
PROBE_EVERY = 10          # chunks (1000 env steps) between isolation-probe evaluations
PROBE_REPLICATES = 8      # stochastic rollouts per probe condition (cheap, for the in-sweep trace;
                          # run a higher-replicate probe separately on the saved final agents for calibration)

N_CONFIGS = 100
N_SEEDS = 5
LHS_SEED = 42

PROBE_KEYS = (
    "edible_approach", "edible_eat_rate", "poison_approach",
    "poison_eat_rate", "approach_disc", "eat_disc",
)

TAG = (('_nosig' if NO_SIGNAL else '')
       + ('_recurrent' if RECURRENT else '')
       + ('_nopainpleasure' if not PAIN_PLEASURE else '')
       + (f'_h{H_SIZE}' if H_SIZE != 5 else ''))
RESULTS_DIR = f"/content/drive/MyDrive/sweep{TAG}"
AGENTS_DIR = f"/content/drive/MyDrive/sweep_agents{TAG}"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AGENTS_DIR, exist_ok=True)


# Generate LHS configs
def generate_configs(n_configs, seed):
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_configs)

    configs = []
    for i, s in enumerate(samples):
        # s is in [0,1]^4, scale each dimension
        mutation_std = float(10 ** (np.log10(0.005) + s[0] * (np.log10(0.2) - np.log10(0.005))))  # log-uniform [0.005, 0.2]
        poison_multiplier = float(-2 - s[1] * 6)  # uniform [-8, -2]
        reprod_cost = float(s[2] * 12)  # was * 25  → [0, 12]
        energy_decay = float(0.05 + s[3] * 0.10)  # was * 0.20 → [0.05, 0.15]

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


def run_one(cfg, seed):
    key = jax.random.key(seed)
    probe_key = jax.random.key(seed + 1_000_003)  # independent stream so probing doesn't perturb the sim trajectory
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
        recurrent=RECURRENT,
        pain_pleasure=PAIN_PLEASURE,
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
        """Isolation-probe the live population: single agent, clean grid, neutral signal.
        Decouples discrimination ability from environment/social confounds."""
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

        # Strict-zero extinction check: alive at end of chunk
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


# Main sweep
configs = generate_configs(N_CONFIGS, LHS_SEED)

# Save the config list once so you have a record
with open(os.path.join(RESULTS_DIR, "configs.json"), "w") as f:
    json.dump(configs, f, indent=2)

total_runs = len(configs) * N_SEEDS
done = 0

for cfg in configs:
    cfg_hash = config_hash(cfg)
    for seed in range(N_SEEDS):
        out_path = os.path.join(RESULTS_DIR, f"cfg{cfg['config_id']:03d}_{cfg_hash}_seed{seed}.json")
        if os.path.exists(out_path):
            done += 1
            continue

        print(f"[{done+1}/{total_runs}] cfg {cfg['config_id']} seed {seed}: "
              f"mut={cfg['mutation_std']:.4f} poison={cfg['poison_multiplier']:.2f} "
              f"reprod={cfg['reprod_cost']:.1f} decay={cfg['energy_decay']:.3f}")

        result, final_agents = run_one(cfg, seed)

        final_window = [d for d in result["chunk_disc"][int(len(result["chunk_disc"]) * 0.8):] if not np.isnan(d)]
        final_disc = np.mean(final_window) if final_window else float('nan')
        final_probe_disc = result["probe_approach_disc"][-1] if result["probe_approach_disc"] else float('nan')
        print(f"  -> final env discrimination: {final_disc:.4f}  final probe discrimination: {final_probe_disc:.4f}")

        if result["extinction_chunk"] is None:
            base = f"cfg{cfg['config_id']:03d}_{cfg_hash}_seed{seed}"
            eqx.tree_serialise_leaves(os.path.join(AGENTS_DIR, f"{base}_agents.eqx"), final_agents.network)
            np.save(os.path.join(AGENTS_DIR, f"{base}_alive.npy"), np.array(final_agents.alive))

        with open(out_path, "w") as f:
            json.dump(result, f)
        done += 1

print(f"\nSweep complete. {total_runs} runs saved to {RESULTS_DIR}")
