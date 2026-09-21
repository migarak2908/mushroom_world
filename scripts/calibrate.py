"""Run a fixed population of identical hand-coded agents (no neural network,
no evolution) under one of the controllers in controllers.py, and log
per-agent consumption events and energy trajectories to a .npz file.

See scripts/calibrate_analysis.py for turning the .npz into tau/f_poison/
g_measured/... statistics, and scripts/calibrate_all.py for the 8-condition
sweep this is built to support.

Energy is logged every step (not subsampled): even the largest configured
run (n_agents=2000, steps=10000) is a (10000, 2000) float32 array, ~80MB,
which is not a real memory concern.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GLOG_minloglevel", "2")

import jax
# Note: a harmless native log line ("Assume version compatibility. PjRt-IFRT
# does not track XLA executable versions") prints once per cached-executable
# load and isn't controllable from Python logging config -- ignore it, it
# doesn't indicate a problem.

# Persistent compilation cache: calibrate_all.py runs each (strategy,
# n_agents, nb_mushrooms, poison_multiplier, epsilon, k) config once per
# seed as an isolated subprocess (needed -- running many distinct shapes in
# one long-lived process accumulates compiled-program memory and can OOM,
# confirmed the hard way). Without a cache, every single seed re-pays full
# compile time from scratch even though 4 of every 5 are shape-identical
# repeats. Caching to disk lets seeds 2-5 of a condition load the already-
# compiled program instead of recompiling -- set CALIBRATE_CACHE_DIR to
# override the location (e.g. a persistent path on Colab/Drive).
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("CALIBRATE_CACHE_DIR", os.path.expanduser("~/.cache/mushroom_jax")),
)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp
import equinox as eqx
import numpy as np

from mushroom_world import MushroomWorld
from controllers import build_policy

STRATEGIES = ("wander", "approach_all", "discriminate")


def run_calibration(strategy, n_agents, nb_mushrooms, grid_size, poison_multiplier,
                     nutrition, energy_decay, energy_start, perc_radius, steps, seed,
                     epsilon=0.05, k=5):
    policy, initial_state_fn = build_policy(strategy, epsilon, k)

    env = MushroomWorld(
        seed=seed, grid_x=grid_size, grid_y=grid_size,
        nb_agents=n_agents, max_agents=n_agents, nb_mushrooms=nb_mushrooms,
        energy_start=energy_start, energy_decay=energy_decay,
        mushroom_nutrition=nutrition, poison_multiplier=poison_multiplier,
        # reproduction is off for this experiment; these fields are unused
        reprod_threshold=energy_start * 2.0, reprod_cost=0.0, mutation_std=0.0,
        # instant regrowth (part 1 of the spec): eaten mushrooms reappear
        # elsewhere the same step they become visible again
        regrowth_period=0,
        frozen_baseline=False,
        signalling=False, consequence_inputs=False,
        allow_death=False, allow_reproduction=False,
        recurrent=False, h_size=5,
    )
    agents, mushrooms = env.reset_fn()
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)
    policy_state0 = initial_state_fn(n_agents)

    @eqx.filter_jit
    def run_scan(key, dynamic_agents, mushrooms, policy_state):
        def step(carry, _):
            key, dynamic_agents, mushrooms, policy_state = carry
            agents = eqx.combine(dynamic_agents, static_agents)
            key, sk = jax.random.split(key)
            agents, mushrooms, _, _, consume_multiplier, policy_state = env.step_fn_policy(
                sk, agents, mushrooms, perc_radius, policy, policy_state
            )
            meal_code = jnp.where(
                consume_multiplier == 1, 1,
                jnp.where(consume_multiplier == poison_multiplier, 2, 0),
            ).astype(jnp.int8)
            dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
            return (key, dynamic_agents, mushrooms, policy_state), (meal_code, agents.energy.astype(jnp.float32))

        return jax.lax.scan(step, (key, dynamic_agents, mushrooms, policy_state), None, length=steps)

    key = jax.random.key(seed)
    _, (meal_codes, energies) = run_scan(key, dynamic_agents, mushrooms, policy_state0)

    meal_codes = np.asarray(meal_codes)   # (steps, n_agents) int8: 0 none, 1 edible, 2 poison
    energies = np.asarray(energies)       # (steps, n_agents) float32

    event_step, event_agent = np.nonzero(meal_codes)
    event_class = meal_codes[event_step, event_agent]

    edible_eaten = (meal_codes == 1).sum(axis=0).astype(np.int32)
    poison_eaten = (meal_codes == 2).sum(axis=0).astype(np.int32)

    return dict(
        energy=energies,
        event_step=event_step.astype(np.int32),
        event_agent=event_agent.astype(np.int32),
        event_class=event_class.astype(np.int8),
        edible_eaten=edible_eaten,
        poison_eaten=poison_eaten,
        strategy=strategy, n_agents=n_agents, nb_mushrooms=nb_mushrooms,
        grid_size=grid_size, poison_multiplier=poison_multiplier,
        nutrition=nutrition, energy_decay=energy_decay, energy_start=energy_start,
        perc_radius=perc_radius, steps=steps, seed=seed, epsilon=epsilon, k=k,
    )


def out_path(out_dir, strategy, n_agents, nb_mushrooms, poison_multiplier, epsilon, k, seed):
    fname = f"{strategy}_n{n_agents}_m{nb_mushrooms}_p{poison_multiplier}_e{epsilon}_k{k}_s{seed}.npz"
    return os.path.join(out_dir, fname)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    p.add_argument("--n_agents", type=int, default=1)
    p.add_argument("--nb_mushrooms", type=int, default=100)
    p.add_argument("--grid_size", type=int, default=100)
    p.add_argument("--poison_multiplier", type=float, default=-1.0)
    p.add_argument("--nutrition", type=float, default=20.0)
    p.add_argument("--energy_decay", type=float, default=0.1)
    p.add_argument("--energy_start", type=float, default=10000.0)
    p.add_argument("--perc_radius", type=int, default=10)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.05,
                    help="per-step probability a controller's movement bits are replaced "
                         "by a uniform random draw; breaks deterministic cycling (0 = off)")
    p.add_argument("--k", type=int, default=5,
                    help="discriminate only: number of blind forward steps to commit to "
                         "after turning away from a detected poison mushroom")
    p.add_argument("--out", type=str, default="./results/calibration/")
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out, exist_ok=True)

    result = run_calibration(
        strategy=args.strategy, n_agents=args.n_agents, nb_mushrooms=args.nb_mushrooms,
        grid_size=args.grid_size, poison_multiplier=args.poison_multiplier,
        nutrition=args.nutrition, energy_decay=args.energy_decay, energy_start=args.energy_start,
        perc_radius=args.perc_radius, steps=args.steps, seed=args.seed, epsilon=args.epsilon,
        k=args.k,
    )

    path = out_path(args.out, args.strategy, args.n_agents, args.nb_mushrooms,
                     args.poison_multiplier, args.epsilon, args.k, args.seed)
    np.savez(path, **result)
    print(f"Saved {path}  "
          f"(events={len(result['event_step'])}, "
          f"edible={int(result['edible_eaten'].sum())}, "
          f"poison={int(result['poison_eaten'].sum())})")


if __name__ == "__main__":
    main()
