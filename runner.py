import jax
import jax.numpy as jnp
import pandas as pd
import wandb

from world import World
from mushrooms import MushroomSource
from food_source import UniformRespawn
from observation import FoodTypeChannel


def build_world(SX, SY, nb_agents, max_agents, energy_start, energy_decay, death_enabled,
                 nb_mushrooms, mushroom_nutrition, poison_proportion, poison_multiplier):
    respawn_strategy = UniformRespawn(SX=SX, SY=SY)
    food_source = MushroomSource(
        SX=SX, SY=SY,
        nb_mushrooms=nb_mushrooms,
        mushroom_nutrition=mushroom_nutrition,
        poison_proportion=poison_proportion,
        poison_multiplier=poison_multiplier,
        respawn_strategy=respawn_strategy,
    )
    channel = FoodTypeChannel(source=food_source, source_idx=0)

    return World(
        seed=0,
        SX=SX, SY=SY,
        nb_agents=nb_agents, max_agents=max_agents,
        energy_start=energy_start, energy_decay=energy_decay,
        food_sources=(food_source,),
        observation_channels=(channel,),
        death_enabled=death_enabled,
    )


def run_policy(world, policy, num_steps, key):
    agents, food_states = world._reset(key)

    def step(carry, _):
        key, agents, food_states = carry
        key, act_key, step_key = jax.random.split(key, 3)
        obs = world._build_obs(agents, food_states)
        turn, move, eat_decision = policy.act(act_key, agents, food_states, obs)
        agents, food_states, eat_delta = world._step(step_key, agents, food_states, turn, move, eat_decision)
        return (key, agents, food_states), eat_delta

    (key, agents, food_states), eat_deltas = jax.lax.scan(step, (key, agents, food_states), None, length=num_steps)

    meals = eat_deltas != 0
    meal_count = meals.sum()
    T = jnp.where(meal_count > 0, num_steps / meal_count, jnp.nan)
    fraction_poisonous = jnp.where(meal_count > 0, (eat_deltas < 0).sum() / meal_count, jnp.nan)
    G = eat_deltas.mean() - world.energy_decay
    return T, fraction_poisonous, G


def run_policy_batch(world, policy, num_steps, num_seeds, base_seed=0):
    keys = jax.vmap(jax.random.key)(jnp.arange(base_seed, base_seed + num_seeds))
    run_one = lambda key: run_policy(world, policy, num_steps, key)
    return jax.vmap(run_one)(keys)   # each output shape (num_seeds,)


def log_results(policy_name, config, T, fraction_poisonous, G):
    wandb.init(project="mushroom-language", config={**config, "policy": policy_name})

    n_zero_meal_seeds = int(jnp.isnan(T).sum())
    T_valid = T[~jnp.isnan(T)]

    wandb.log({
        "T_mean": float(jnp.nanmean(T)), "T_std": float(jnp.nanstd(T)),
        "fraction_poisonous_mean": float(jnp.nanmean(fraction_poisonous)),
        "G_mean": float(G.mean()), "G_std": float(G.std()),
        "zero_meal_seeds": n_zero_meal_seeds,
        "T_histogram": wandb.Histogram(T_valid) if T_valid.size > 0 else None,
        "G_histogram": wandb.Histogram(G),
        "per_seed": wandb.Table(dataframe=pd.DataFrame({
            "seed": range(len(T)),
            "T": T, "fraction_poisonous": fraction_poisonous, "G": G,
        })),
    })
    wandb.finish()
