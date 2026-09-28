import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from policy import wander
from runner import build_world, run_policy_batch, log_results


SX = 20
SY = 20
NB_AGENTS = 1
MAX_AGENTS = 1
ENERGY_START = 200.0
ENERGY_DECAY = 0.5
DEATH_ENABLED = False

NB_MUSHROOMS = 16          # m = NB_MUSHROOMS / (SX*SY) = 0.04
MUSHROOM_NUTRITION = 10.0  # N
POISON_PROPORTION = 0.4    # q
POISON_MULTIPLIER = 1.0    # P

NUM_STEPS = 20000
NUM_SEEDS = 1000


if __name__ == "__main__":
    world = build_world(
        SX=SX, SY=SY, nb_agents=NB_AGENTS, max_agents=MAX_AGENTS,
        energy_start=ENERGY_START, energy_decay=ENERGY_DECAY, death_enabled=DEATH_ENABLED,
        nb_mushrooms=NB_MUSHROOMS, mushroom_nutrition=MUSHROOM_NUTRITION,
        poison_proportion=POISON_PROPORTION, poison_multiplier=POISON_MULTIPLIER,
    )
    T, fraction_poisonous, G = run_policy_batch(world, wander, NUM_STEPS, NUM_SEEDS)

    m = NB_MUSHROOMS / (SX * SY)
    n_zero_meal = int(jnp.isnan(T).sum())
    print(f"m = {m}")
    print(f"T:  measured mean = {jnp.nanmean(T):.2f} (predicted: between 4/m = {4/m:.2f} and the random-walk cover-time bound)")
    print(f"fraction poisonous: measured mean = {jnp.nanmean(fraction_poisonous):.3f} (predicted ~= q = {POISON_PROPORTION})")
    print(f"G:  measured mean = {G.mean():.4f}")
    print(f"zero-meal seeds: {n_zero_meal}/{NUM_SEEDS}")

    config = dict(
        SX=SX, SY=SY, m=m, q=POISON_PROPORTION, N=MUSHROOM_NUTRITION, P=POISON_MULTIPLIER,
        d=ENERGY_DECAY, energy_start=ENERGY_START, num_steps=NUM_STEPS, num_seeds=NUM_SEEDS,
    )
    log_results("wander", config, T, fraction_poisonous, G)
