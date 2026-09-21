"""Diagnostic harness for the discriminate controller's avoidance rule
(currently: memory-based turn-then-commit, see controllers.make_discriminate).
Runs a lone agent, jitted end to end via lax.scan, then post-processes the
per-step trace in plain numpy to report:
    max_consecutive_avoid_steps  -- expect approximately k+1 (one turn step
                                     plus k blind forward steps), not hundreds
    n_reacquisitions             -- transitions poisonous-visible -> hidden
                                     -> back to the SAME mushroom index while
                                     still poisonous; expect near zero
    unique_positions_visited     -- should keep climbing, not plateau
    msd_from_start                -- mean squared toroidal displacement from
                                     the step-0 position; should grow like
                                     approach_all's, not stay bounded near a
                                     small value (that was the outward-spiral
                                     design's failure mode: avoidance is
                                     repulsive, so a poison mushroom between
                                     the agent and unexplored territory acts
                                     as a one-way barrier that pushes it back)
    meals, meals_per_1000

This bypasses step_fn_policy to read privileged environment state (the
avoided mushroom's true index/distance) for diagnostics only -- the
controller itself still only ever sees `obs` (and, now, its own state).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from mushroom_world import MushroomWorld
from controllers import build_policy, POISON_PROTOTYPE


def _msd_from_start(posx, posy, grid_size):
    dx = (posx - posx[0] + grid_size // 2) % grid_size - grid_size // 2
    dy = (posy - posy[0] + grid_size // 2) % grid_size - grid_size // 2
    return dx.astype(np.int64) ** 2 + dy.astype(np.int64) ** 2


def run_diagnostics(nb_mushrooms, steps=10000, grid_size=100, poison_multiplier=-1.0,
                     nutrition=20.0, energy_decay=0.1, energy_start=10000.0, perc_radius=10,
                     seed=0, epsilon=0.05, k=5, debug=False):
    env = MushroomWorld(
        seed=seed, grid_x=grid_size, grid_y=grid_size, nb_agents=1, max_agents=1,
        nb_mushrooms=nb_mushrooms, energy_start=energy_start, energy_decay=energy_decay,
        mushroom_nutrition=nutrition, poison_multiplier=poison_multiplier,
        reprod_threshold=energy_start * 2.0, reprod_cost=0.0, mutation_std=0.0,
        regrowth_period=0, frozen_baseline=False,
        signalling=False, consequence_inputs=False,
        allow_death=False, allow_reproduction=False, recurrent=False, h_size=5,
    )
    agents, mushrooms = env.reset_fn()
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)
    policy, initial_state_fn = build_policy("discriminate", epsilon, k)
    policy_state0 = initial_state_fn(1)

    def nearest_and_dist(agents, mushrooms):
        x_diff = (agents.posx[:, None] - mushrooms.posx[None, :] + grid_size // 2) % grid_size - grid_size // 2
        y_diff = (agents.posy[:, None] - mushrooms.posy[None, :] + grid_size // 2) % grid_size - grid_size // 2
        dist_sq = (x_diff ** 2 + y_diff ** 2).astype(jnp.float32)
        present = mushrooms.regrowth_countdown == 0
        dist_sq = jnp.where(present[None, :], dist_sq, jnp.inf)
        nearest = jnp.argmin(dist_sq, axis=1)
        dist = jnp.sqrt(dist_sq[jnp.arange(1), nearest])
        return nearest[0], dist[0]

    @eqx.filter_jit
    def run_scan(key, dynamic_agents, mushrooms, policy_state):
        def step(carry, _):
            key, dynamic_agents, mushrooms, policy_state = carry
            agents = eqx.combine(dynamic_agents, static_agents)
            nearest_idx, dist = nearest_and_dist(agents, mushrooms)

            key, sk1, sk2, sk3 = jax.random.split(key, 4)
            obs = env._compute_obs(sk1, agents, mushrooms, perc_radius)
            feats = obs[0, 3:13]
            hidden = jnp.all(feats == 0.5)
            hamming = jnp.sum(jnp.abs(feats - POISON_PROTOTYPE))
            is_poison_nearby = (~hidden) & (hamming <= 1)

            state_row = jax.tree_util.tree_map(lambda s: s[0], policy_state)
            action, new_state_row = policy(obs[0], sk2, state_row)
            new_state = jax.tree_util.tree_map(lambda s: s[None], new_state_row)

            agents, mushrooms, _, _, consume_multiplier = env._compute_update(
                sk3, action[None, :], agents, mushrooms
            )

            dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
            out = (agents.posx[0], agents.posy[0], nearest_idx, dist,
                   is_poison_nearby, consume_multiplier[0])
            return (key, dynamic_agents, mushrooms, new_state), out

        return jax.lax.scan(step, (key, dynamic_agents, mushrooms, policy_state), None, length=steps)

    key = jax.random.key(seed)
    _, (posx, posy, nearest_idx, dist, is_poison_nearby, meal_class) = run_scan(
        key, dynamic_agents, mushrooms, policy_state0
    )
    posx, posy = np.asarray(posx), np.asarray(posy)
    nearest_idx = np.asarray(nearest_idx)
    dist = np.asarray(dist)
    is_poison_nearby = np.asarray(is_poison_nearby)
    meal_class = np.asarray(meal_class)

    positions = set(zip(posx.tolist(), posy.tolist()))
    meals = int((meal_class != 0).sum())
    msd = _msd_from_start(posx, posy, grid_size)

    max_consecutive_avoid = 0
    consecutive = 0
    avoid_target = None
    last_avoid_target = None
    reacquisitions = 0
    dist_window = []
    violations = []

    for t in range(steps):
        if is_poison_nearby[t]:
            idx = int(nearest_idx[t])
            if avoid_target != idx:
                if last_avoid_target is not None and idx == last_avoid_target:
                    reacquisitions += 1
                avoid_target = idx
                consecutive = 1
                dist_window = [float(dist[t])]
            else:
                consecutive += 1
                dist_window.append(float(dist[t]))
                if debug and len(dist_window) >= 5 and dist_window[-1] < dist_window[-5] - 1e-6:
                    violations.append((t, tuple(dist_window[-5:])))
            max_consecutive_avoid = max(max_consecutive_avoid, consecutive)
        else:
            if avoid_target is not None:
                last_avoid_target = avoid_target
            avoid_target = None
            consecutive = 0
            dist_window = []

    result = dict(
        nb_mushrooms=nb_mushrooms, steps=steps, epsilon=epsilon, k=k,
        max_consecutive_avoid_steps=max_consecutive_avoid,
        n_reacquisitions=reacquisitions,
        unique_positions_visited=len(positions),
        meals=meals, meals_per_1000=meals / steps * 1000,
        msd_from_start_final=int(msd[-1]),
        msd_from_start_mean=float(msd.mean()),
    )
    if debug:
        result["distance_violations"] = violations
    return result, msd, posx, posy


def run_baseline(strategy_fn, nb_mushrooms, steps=10000, grid_size=100, poison_multiplier=-1.0,
                  nutrition=20.0, energy_decay=0.1, energy_start=10000.0, perc_radius=10,
                  seed=0, epsilon=0.05):
    """Run any stateless controller (approach_all) -- used here for comparison."""
    from controllers import epsilon_greedy
    env = MushroomWorld(
        seed=seed, grid_x=grid_size, grid_y=grid_size, nb_agents=1, max_agents=1,
        nb_mushrooms=nb_mushrooms, energy_start=energy_start, energy_decay=energy_decay,
        mushroom_nutrition=nutrition, poison_multiplier=poison_multiplier,
        reprod_threshold=energy_start * 2.0, reprod_cost=0.0, mutation_std=0.0,
        regrowth_period=0, frozen_baseline=False,
        signalling=False, consequence_inputs=False,
        allow_death=False, allow_reproduction=False, recurrent=False, h_size=5,
    )
    agents, mushrooms = env.reset_fn()
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)
    policy = epsilon_greedy(strategy_fn, epsilon)
    policy_state0 = jnp.zeros((1,), dtype=jnp.int32)

    @eqx.filter_jit
    def run_scan(key, dynamic_agents, mushrooms, policy_state):
        def step(carry, _):
            key, dynamic_agents, mushrooms, policy_state = carry
            agents = eqx.combine(dynamic_agents, static_agents)
            key, sk = jax.random.split(key)
            agents, mushrooms, _, _, cm, policy_state = env.step_fn_policy(
                sk, agents, mushrooms, perc_radius, policy, policy_state
            )
            dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
            return (key, dynamic_agents, mushrooms, policy_state), (agents.posx[0], agents.posy[0], cm[0])

        return jax.lax.scan(step, (key, dynamic_agents, mushrooms, policy_state), None, length=steps)

    key = jax.random.key(seed)
    _, (posx, posy, meal_class) = run_scan(key, dynamic_agents, mushrooms, policy_state0)
    posx, posy, meal_class = np.asarray(posx), np.asarray(posy), np.asarray(meal_class)
    meals = int((meal_class != 0).sum())
    positions = set(zip(posx.tolist(), posy.tolist()))
    msd = _msd_from_start(posx, posy, grid_size)
    return dict(meals=meals, meals_per_1000=meals / steps * 1000,
                unique_positions_visited=len(positions),
                msd_from_start_final=int(msd[-1]), msd_from_start_mean=float(msd.mean()))


if __name__ == "__main__":
    from controllers import approach_all

    EPSILON = 0.05
    K = 5

    for nb in (100, 1000):
        d, msd, posx, posy = run_diagnostics(nb_mushrooms=nb, steps=10000, seed=0,
                                              epsilon=EPSILON, k=K, debug=True)
        base = run_baseline(approach_all, nb_mushrooms=nb, steps=10000, seed=0, epsilon=EPSILON)
        print(f"nb_mushrooms={nb:5d}  meals/1000={d['meals_per_1000']:7.2f}  "
              f"approach_all_baseline/1000={base['meals_per_1000']:7.2f}  "
              f"ratio={d['meals_per_1000']/base['meals_per_1000']:.3f}")
        print(f"  max_consec_avoid={d['max_consecutive_avoid_steps']:4d} (expect ~{K+1})  "
              f"reacquisitions={d['n_reacquisitions']:3d}  "
              f"violations={len(d['distance_violations'])}")
        print(f"  unique_positions: discriminate={d['unique_positions_visited']:5d}  "
              f"approach_all={base['unique_positions_visited']:5d}")
        print(f"  msd_from_start (mean): discriminate={d['msd_from_start_mean']:8.1f}  "
              f"approach_all={base['msd_from_start_mean']:8.1f}")
        print()
