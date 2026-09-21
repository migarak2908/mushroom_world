"""Hand-coded controllers for the calibration experiment (scripts/calibrate.py).

Uniform calling convention for mushroom_world.step_fn_policy:
`policy(obs, key, state) -> (actions, new_state)`, vmappable per agent.
wander/approach_all are stateless (state passed through unchanged -- see
epsilon_greedy). discriminate carries a minimal two-field state (escaping,
steps_left) -- see make_discriminate. Assumes the calibration configuration:
signalling=False, consequence_inputs=False, so obs is the 13-dim
[cos, sin, dist, features(10)] layout (mushroom_world.ObsLayout) and actions
are the 3-dim [move_hi, move_lo, eat] layout (mushroom_world.ActionLayout).

Turn semantics (verified against mushroom_world._compute_obs / _compute_update):
obs[cos], obs[sin] give the bearing to the nearest mushroom in the agent's
own egocentric frame -- dtheta = arctan2(sin, cos), 0 = straight ahead. This
is NOT a raw world-frame angle; _compute_obs rotates it into the agent's
current heading, which is otherwise never exposed in the observation.
Stepping through _compute_update confirms:
    move_idx=3 (bits 1,1): move forward, dtheta unchanged
    move_idx=2 (bits 1,0): turn, dtheta -> dtheta - 90 deg
    move_idx=1 (bits 0,1): turn, dtheta -> dtheta + 90 deg
So to reduce |dtheta| (approach): use move_idx=2 when dtheta > 0, move_idx=1
when dtheta < 0. To increase |dtheta| (turn away, see make_discriminate):
the mirror image -- move_idx=1 when dtheta > 0, move_idx=2 when dtheta < 0.
Ties (dtheta == 0 or +/-pi) are broken toward the approach-left / avoid-right
branch by the >= 0 convention below.

obs[dist] is already normalised by perc_radius (mushroom_world._compute_obs:
clip(dist_to_mush / perc_radius, 0, 1)).

Prior discriminate designs (flee-with-per-step-jitter, orbit, outward
spiral) were all fully deterministic or only locally randomised, and in a
fully deterministic environment (fixed mushroom layout, no other agents) a
deterministic policy has a finite state space and must eventually cycle.
epsilon_greedy below adds the same kind of stochasticity the NN path
already gets from sampling its bernoulli outputs every step, and the spiral
design fixed cycling within one avoidance episode -- but empirically
(scripts/calibrate_diagnostics.py, scripts/calibrate_density_sweep.py) the
overall trajectory still stayed confined to a small fraction of the map at
every density and epsilon tested: mean-squared-displacement from the start
position never grew the way approach_all's does, because avoidance is
inherently repulsive -- a poison mushroom between the agent and unexplored
territory pushes it back the way it came rather than letting it pass. The
memory-based design below (turn away once, then commit to k blind forward
steps) is a qualitatively different mechanism: a single decisive turn plus
a fixed run of forward motion, rather than a per-step reactive rule that
can get recomputed into a stall every step.
"""
import jax
import jax.numpy as jnp

from mushroom_world import ObsLayout, mushroom_prototype

_LAYOUT = ObsLayout(signalling=False, consequence_inputs=False)

FORWARD = jnp.array([1, 1], dtype=jnp.int32)     # move_idx=3
TURN_LEFT = jnp.array([1, 0], dtype=jnp.int32)   # move_idx=2, dtheta -= 90 deg
TURN_RIGHT = jnp.array([0, 1], dtype=jnp.int32)  # move_idx=1, dtheta += 90 deg
EAT = jnp.array(1, dtype=jnp.int32)

FORWARD_THRESHOLD = jnp.pi / 4

# poison prototype, per mushroom_world.reset_fn: mushroom_type == 1 (edible)
# draws features from MUSH_LIBRARY[10:20] (variants of mushroom_prototype[1]);
# poison draws from MUSH_LIBRARY[0:10] (variants of mushroom_prototype[0]).
POISON_PROTOTYPE = mushroom_prototype[0].astype(jnp.float32)  # 0 0 0 0 0 1 1 1 1 1


def _approach_move(dtheta):
    return jnp.where(
        jnp.abs(dtheta) <= FORWARD_THRESHOLD,
        FORWARD,
        jnp.where(dtheta >= 0, TURN_LEFT, TURN_RIGHT),
    )


def _avoid_turn(dtheta):
    """Mirror image of _approach_move's turn branch: turns AWAY from the
    target (increases |dtheta|, heading toward facing directly away) rather
    than toward it. Never returns FORWARD -- avoidance always starts with
    exactly one turn (see make_discriminate)."""
    return jnp.where(dtheta > 0, TURN_RIGHT, TURN_LEFT)


def wander(obs, key):
    """Uniform random action every step. Stateless."""
    return jax.random.bernoulli(key, 0.5, (3,)).astype(jnp.int32)


def approach_all(obs, key):
    """Turn/move toward the nearest mushroom regardless of class; always
    attempts to eat (the environment only lets this succeed while on a
    present mushroom's cell). Ignores the feature channel entirely.
    Stateless."""
    dtheta = jnp.arctan2(obs[_LAYOUT.sin], obs[_LAYOUT.cos])
    move = _approach_move(dtheta)
    return jnp.concatenate([move, EAT[None]])


def make_discriminate(k=5):
    """Build a discriminate controller with escape length k.

    Minimal state: (escaping: int32 0/1, steps_left: int32), both start at
    (0, 0). Each call:
      - steps_left > 0: ignore the observation entirely, emit "move
        forward", decrement steps_left. This is the memory: once committed
        to escaping, the agent does not re-evaluate the threat every step
        (which is what let every previous reactive design get recomputed
        into a stall) -- it just leaves.
      - steps_left == 0: normal rules. Features hidden -> approach (rule
        1). Features visible and Hamming distance to the poison prototype
        <= 1 -> turn away (the smaller-turn / mirror-image rule above),
        set steps_left = k. Features visible and not poisonous -> approach
        (rule 3, same as rule 1).

    The turn-then-commit step and the following k forward steps together
    form one avoidance episode of k+1 actions; diagnostics
    (max_consecutive_avoid_steps in scripts/calibrate_diagnostics.py,
    measured independently from the observation, not from this state)
    should read approximately k+1."""

    def discriminate(obs, state):
        escaping, steps_left = state

        def continue_escape(_):
            # never eat while blindly escaping -- the observation is ignored
            # here entirely (including for eat), so there is no way to
            # confirm anything crossed during this run is safe; this is the
            # actual fix for the poison-eating problem (100% of it traced to
            # this branch's previously-unconditional eat=1, not the trigger
            # step) -- see controllers.py history / calibrate_diagnostics.py.
            new_steps_left = steps_left - 1
            no_eat = jnp.array(0, dtype=jnp.int32)
            action = jnp.concatenate([FORWARD, no_eat[None]])
            new_state = ((new_steps_left > 0).astype(jnp.int32), new_steps_left)
            return action, new_state

        def evaluate(_):
            feats = obs[_LAYOUT.features]
            features_hidden = jnp.all(feats == 0.5)
            hamming = jnp.sum(jnp.abs(feats - POISON_PROTOTYPE))
            is_poison_nearby = (~features_hidden) & (hamming <= 1)

            dtheta = jnp.arctan2(obs[_LAYOUT.sin], obs[_LAYOUT.cos])
            avoid_move = _avoid_turn(dtheta)
            approach_move = _approach_move(dtheta)
            move = jnp.where(is_poison_nearby, avoid_move, approach_move)
            # do not eat on the very step poison is detected: eat is resolved
            # on the pre-move position, so if the agent is already co-located
            # with this mushroom (e.g. having just landed on/near it during a
            # prior blind escape run), an unconditional eat would consume it
            # this same step regardless of the avoid-turn just chosen -- the
            # turn only affects movement, not this step's consumption.
            eat = jnp.where(is_poison_nearby, 0, EAT).astype(jnp.int32)
            action = jnp.concatenate([move, eat[None]])

            new_steps_left = jnp.where(is_poison_nearby, k, 0)
            new_state = ((new_steps_left > 0).astype(jnp.int32), new_steps_left)
            return action, new_state

        return jax.lax.cond(steps_left > 0, continue_escape, evaluate, operand=None)

    return discriminate


def initial_discriminate_state(n_agents):
    return (jnp.zeros((n_agents,), dtype=jnp.int32), jnp.zeros((n_agents,), dtype=jnp.int32))


def epsilon_greedy(policy, epsilon):
    """Wrap a stateless controller `(obs, key) -> actions` into the uniform
    `(obs, key, state) -> (actions, state)` convention (state passed through
    unchanged), adding: with probability epsilon each step, its movement
    bits (not the eat bit) are replaced by a uniform draw from the 4-member
    movement action set {stay, turn_left, turn_right, forward} --
    equivalently, each movement bit independently Bernoulli(0.5). epsilon=0
    reproduces the underlying controller exactly."""
    def wrapped(obs, key, state):
        k_policy, k_explore, k_random = jax.random.split(key, 3)
        preferred = policy(obs, k_policy)
        explore = jax.random.bernoulli(k_explore, epsilon)
        random_move = jax.random.bernoulli(k_random, 0.5, (2,)).astype(jnp.int32)
        move = jnp.where(explore, random_move, preferred[:2])
        action = jnp.concatenate([move, preferred[2:]])
        return action, state
    return wrapped


def epsilon_greedy_stateful(policy, epsilon):
    """Same as epsilon_greedy, but for a stateful controller already in the
    `(obs, state) -> (actions, state)` form (i.e. make_discriminate's
    output) -- adds the key argument needed for the exploration noise."""
    def wrapped(obs, key, state):
        preferred, new_state = policy(obs, state)
        k_explore, k_random = jax.random.split(key)
        explore = jax.random.bernoulli(k_explore, epsilon)
        random_move = jax.random.bernoulli(k_random, 0.5, (2,)).astype(jnp.int32)
        move = jnp.where(explore, random_move, preferred[:2])
        action = jnp.concatenate([move, preferred[2:]])
        return action, new_state
    return wrapped


def build_policy(strategy, epsilon, k=5):
    """Returns (policy, initial_state_fn) where policy has the uniform
    `(obs, key, state) -> (actions, state)` signature and
    initial_state_fn(n_agents) builds that policy's initial per-agent
    state pytree."""
    if strategy == "wander":
        return epsilon_greedy(wander, epsilon), lambda n: jnp.zeros((n,), dtype=jnp.int32)
    elif strategy == "approach_all":
        return epsilon_greedy(approach_all, epsilon), lambda n: jnp.zeros((n,), dtype=jnp.int32)
    elif strategy == "discriminate":
        return epsilon_greedy_stateful(make_discriminate(k), epsilon), initial_discriminate_state
    raise ValueError(f"unknown strategy {strategy!r}")


CONTROLLERS = {"wander": wander, "approach_all": approach_all}
