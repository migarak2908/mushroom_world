import jax
import jax.numpy as jnp

from world import World
from mushrooms import MushroomSource
from food_source import UniformRespawn
from observation import FoodTypeChannel


def build_world(SX, SY, nb_agents, max_agents, energy_start, energy_decay, death_enabled,
                reproduction_enabled, r_thresh, r_cost,
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
        reproduction_enabled=reproduction_enabled,
        r_thresh=r_thresh,
        r_cost=r_cost
    )


def run_policy(world, policy, num_steps, key):
    agents, food_states = world._reset(key)

    def step(carry, _):
        key, agents, food_states = carry
        key, act_key, step_key = jax.random.split(key, 3)
        obs = world._build_obs(agents, food_states)
        turn, move, eat_decision = policy.act(act_key, agents, food_states, obs)
        agents, food_states, eat_delta, births = world._step(step_key, agents, food_states, turn, move, eat_decision)
        return (key, agents, food_states), (births, agents.alive, eat_delta)

    (key, agents, food_states), (births, alive, eat_deltas) = jax.lax.scan(
        step, (key, agents, food_states), None, length=num_steps)

    return births, alive, eat_deltas


def run_policy_batch(world, policy, num_steps, num_seeds, base_seed=0):
    keys = jax.vmap(jax.random.key)(jnp.arange(base_seed, base_seed + num_seeds))
    run_one = lambda key: run_policy(world, policy, num_steps, key)
    return jax.vmap(run_one)(keys)   # each output shape (num_seeds, num_steps, nb_agents)
