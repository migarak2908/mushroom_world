import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import jax.numpy as jnp
import pandas as pd

from baselines import wander, approach_all, discriminating_wanderer, discriminate, wander_0_5, approach_all_0_5
from runner import build_world, run_policy_batch
from metrics import compute_metrics, log_results


SX, SY = 20, 20
NB_AGENTS, MAX_AGENTS = 1, 1
ENERGY_START = 200.0
ENERGY_DECAY = 0.5
MUSHROOM_NUTRITION = 10.0   # N -- reference value, matches the original Stage 1 sweep
POISON_MULTIPLIER = 1.0    # P
NUM_STEPS = 20000
NUM_SEEDS = 1000

M_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
Q_VALUES = [0.25, 0.5, 0.75]

POLICIES = [
    ("wander", wander),
    ("approach_all", approach_all),
    ("discriminating_wanderer", discriminating_wanderer),
    ("discriminate", discriminate),
    ("wander_0_5", wander_0_5),
    ("approach_all_0_5", approach_all_0_5),
]


if __name__ == "__main__":
    rows = []

    for m, q in itertools.product(M_VALUES, Q_VALUES):
        nb_mushrooms = round(m * SX * SY)

        world = build_world(
            SX=SX, SY=SY, nb_agents=NB_AGENTS, max_agents=MAX_AGENTS,
            energy_start=ENERGY_START, energy_decay=ENERGY_DECAY, death_enabled=False,
            reproduction_enabled=False, r_thresh=0.0, r_cost=0.0,
            nb_mushrooms=nb_mushrooms, mushroom_nutrition=MUSHROOM_NUTRITION,
            poison_proportion=q, poison_multiplier=POISON_MULTIPLIER,
        )
        config = dict(SX=SX, SY=SY, m=m, q=q, N=MUSHROOM_NUTRITION, P=POISON_MULTIPLIER,
                      d=ENERGY_DECAY, energy_start=ENERGY_START, num_steps=NUM_STEPS, num_seeds=NUM_SEEDS)

        for name, policy in POLICIES:
            births, alive, eat_deltas = run_policy_batch(world, policy, NUM_STEPS, NUM_SEEDS)
            T, T_first, fraction_poisonous, G = compute_metrics(eat_deltas, NUM_STEPS, ENERGY_DECAY)

            summary = log_results(name, config, T, T_first, fraction_poisonous, G)

            print(f"{name:25s} m={m} q={q}  T={summary['T_mean']:.2f}  T_first={summary['T_first_mean']:.2f}  "
                  f"frac_poison={summary['fraction_poisonous_mean']:.3f}  G={summary['G_mean']:.4f}  "
                  f"zero_meal={summary['zero_meal_seeds']}/{NUM_SEEDS}")

            rows.append(dict(policy=name, m=m, q=q, **summary))

    df = pd.DataFrame(rows)
    print("\n===== CSV SUMMARY (copy from here) =====")
    print(df.to_csv(index=False))
