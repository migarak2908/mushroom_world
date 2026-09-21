"""Epsilon sweep to pick the smallest epsilon at which neither approach_all
nor discriminate traps: strategy in {approach_all, discriminate} x
epsilon in {0, 0.02, 0.05, 0.1}, 5 seeds, 1 agent, 1000 mushrooms.
Reports meals per 1000 steps and unique_positions_visited for each cell,
mean +/- std over seeds. eps=0 is expected to reproduce trapping (very low
meals, very low unique_positions); this documents the deterministic-
trapping problem rather than hiding it, and the result picks --epsilon for
the main calibration.

Writes results/calibration/epsilon_sweep.csv (one row per
(strategy, epsilon, seed)); state which for --out via CLI if needed.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from mushroom_world import MushroomWorld
from controllers import CONTROLLERS, epsilon_greedy

NB_MUSHROOMS = 1000
N_AGENTS = 1
GRID_SIZE = 100
POISON_MULTIPLIER = -1.0
NUTRITION = 20.0
ENERGY_DECAY = 0.1
ENERGY_START = 10000.0
PERC_RADIUS = 10
STEPS = 10000
SEEDS = range(5)
EPSILONS = (0.0, 0.02, 0.05, 0.1)
STRATEGIES = ("approach_all", "discriminate")
OUT_PATH = "./results/calibration/epsilon_sweep.csv"


def build_runner(strategy, epsilon):
    policy = epsilon_greedy(CONTROLLERS[strategy], epsilon)
    env = MushroomWorld(
        seed=0, grid_x=GRID_SIZE, grid_y=GRID_SIZE, nb_agents=N_AGENTS, max_agents=N_AGENTS,
        nb_mushrooms=NB_MUSHROOMS, energy_start=ENERGY_START, energy_decay=ENERGY_DECAY,
        mushroom_nutrition=NUTRITION, poison_multiplier=POISON_MULTIPLIER,
        reprod_threshold=ENERGY_START * 2.0, reprod_cost=0.0, mutation_std=0.0,
        regrowth_period=0, frozen_baseline=False,
        signalling=False, consequence_inputs=False,
        allow_death=False, allow_reproduction=False, recurrent=False, h_size=5,
    )

    @eqx.filter_jit
    def run_seed(seed):
        agents, mushrooms = eqx.tree_at(lambda e: e.seed, env, seed).reset_fn()
        dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)

        def step(carry, _):
            key, dynamic_agents, mushrooms = carry
            agents = eqx.combine(dynamic_agents, static_agents)
            key, sk = jax.random.split(key)
            agents, mushrooms, _, _, cm = env.step_fn_policy(sk, agents, mushrooms, PERC_RADIUS, policy)
            dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
            return (key, dynamic_agents, mushrooms), (agents.posx[0], agents.posy[0], cm[0])

        key = jax.random.key(seed)
        _, (posx, posy, meal_class) = jax.lax.scan(
            step, (key, dynamic_agents, mushrooms), None, length=STEPS
        )
        return posx, posy, meal_class

    return run_seed


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rows = []

    for strategy in STRATEGIES:
        for epsilon in EPSILONS:
            run_seed = build_runner(strategy, epsilon)
            for seed in SEEDS:
                posx, posy, meal_class = run_seed(seed)
                posx, posy, meal_class = np.asarray(posx), np.asarray(posy), np.asarray(meal_class)
                meals = int((meal_class != 0).sum())
                unique_positions = len(set(zip(posx.tolist(), posy.tolist())))
                row = dict(strategy=strategy, epsilon=epsilon, seed=seed,
                           meals=meals, meals_per_1000=meals / STEPS * 1000,
                           unique_positions_visited=unique_positions)
                rows.append(row)
                print(f"{strategy:13s} eps={epsilon:.2f} seed={seed}  "
                      f"meals/1000={row['meals_per_1000']:7.2f}  unique_positions={unique_positions:5d}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {OUT_PATH}")

    print("\n=== summary (mean +/- std over 5 seeds) ===")
    for strategy in STRATEGIES:
        for epsilon in EPSILONS:
            cell = [r for r in rows if r["strategy"] == strategy and r["epsilon"] == epsilon]
            m = np.array([r["meals_per_1000"] for r in cell])
            u = np.array([r["unique_positions_visited"] for r in cell])
            print(f"{strategy:13s} eps={epsilon:.2f}  "
                  f"meals/1000={m.mean():7.2f}+/-{m.std():5.2f}  "
                  f"unique_positions={u.mean():7.1f}+/-{u.std():5.1f}")


if __name__ == "__main__":
    main()
