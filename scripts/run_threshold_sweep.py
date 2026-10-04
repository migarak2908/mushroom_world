import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from baselines import wander, approach_all, discriminating_wanderer, discriminate, wander_0_5, approach_all_0_5
from runner import build_world, run_policy_batch
from metrics import compute_reproduction_metrics, summarize_condition


SX, SY = 20, 20
NB_AGENTS, MAX_AGENTS = 1, 1
ENERGY_START = 200.0
ENERGY_DECAY = 0.1
MUSHROOM_NUTRITION = 62  # N -- reference value, matches the original Stage 1 sweep
POISON_MULTIPLIER = 0.5    # P
NUM_STEPS = 100000
NUM_SEEDS = 1000
M = 0.04
Q = 0.5
R_THRESHOLD = [250, 300, 350, 400, 450]
R_COST = 200.0


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

    for thresh in R_THRESHOLD:

        nb_mushrooms = round(M * SX * SY)

        world = build_world(
            SX=SX, SY=SY, nb_agents=NB_AGENTS, max_agents=MAX_AGENTS,
            energy_start=ENERGY_START, energy_decay=ENERGY_DECAY, death_enabled=True,
            reproduction_enabled=True, r_thresh=thresh, r_cost=R_COST,
            nb_mushrooms=nb_mushrooms, mushroom_nutrition=MUSHROOM_NUTRITION,
            poison_proportion=Q, poison_multiplier=POISON_MULTIPLIER,
        )
        for name, policy in POLICIES:
            births, alive, eat_deltas = run_policy_batch(world, policy, NUM_STEPS, NUM_SEEDS)
            metrics = compute_reproduction_metrics(births, alive, eat_deltas, NUM_STEPS)
            summary = summarize_condition(metrics)

            print(f"{name:25s} r_thresh={thresh}  offspring_mean={summary['offspring_mean']:.3f}  "
                  f"offspring_p90={summary['offspring_p90']:.1f}  lifespan_mean={summary['lifespan_mean']:.1f}  "
                  f"censored_frac={summary['censored_frac']:.3f}")

            rows.append(dict(policy=name, r_thresh=thresh, m=M, q=Q, **summary))

    df = pd.DataFrame(rows)
    print("\n===== CSV SUMMARY (copy from here) =====")
    print(df.to_csv(index=False))
