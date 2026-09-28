from mushroom_world import SIGNALS, MUSH_LIBRARY, DX, DY, TURN, MOVE
from agent import Network, Recurrent_Network
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

PROBE_GRID = 20
AGENT_X, AGENT_Y = 10, 10
MUSH_X, MUSH_Y = 10, 18       # distance 8, always within perc_radius=10
INITIAL_DIST = float(MUSH_Y - AGENT_Y)
PROBE_STEPS = 40
PERC_RADIUS = 10
MAX_AGENTS = 5000
PROBE_REPLICATES = 8  # default stochastic rollouts per (agent, direction, feature) condition

NEUTRAL_SIGNAL = SIGNALS[-1]  # [0.5, 0.5, 0.5] — matches no_signal training condition
NEUTRAL_OUTCOME = jnp.zeros(2)  # [pain, pleasure] — no consumption event in this synthetic probe step


def run_probe_trial(network, direction, feature_idx, key):
    """Single stochastic probe trial. Returns (approach_score, ate_any).

    Actions are sampled with jax.random.bernoulli, matching how the real
    environment (mushroom_world.step_fn) selects actions from the network's
    sigmoid outputs. Rounding to the nearest action instead would erase any
    bias too weak to cross the 0.5 threshold, even though that same bias is
    exactly what the environment's stochastic sampling picks up on at scale.
    """
    px = jnp.array(AGENT_X, dtype=jnp.int32)
    py = jnp.array(AGENT_Y, dtype=jnp.int32)
    d  = jnp.array(direction, dtype=jnp.int32)
    mx = jnp.array(MUSH_X, dtype=jnp.int32)
    my = jnp.array(MUSH_Y, dtype=jnp.int32)
    features = MUSH_LIBRARY[feature_idx]

    def dist_to_mush(px, py):
        xd = (px - mx + PROBE_GRID // 2) % PROBE_GRID - PROBE_GRID // 2
        yd = (py - my + PROBE_GRID // 2) % PROBE_GRID - PROBE_GRID // 2
        return jnp.sqrt((xd ** 2 + yd ** 2).astype(jnp.float32) + 1e-8), xd, yd

    def step(carry, _):
        px, py, d, hidden, key = carry
        key, subkey = jax.random.split(key)

        dist, xd, yd = dist_to_mush(px, py)
        cos_ = xd.astype(jnp.float32) / dist
        sin_ = yd.astype(jnp.float32) / dist
        feat_obs = jnp.where(dist <= PERC_RADIUS, features, MUSH_LIBRARY[20])
        obs = jnp.concat([cos_[None], sin_[None], feat_obs, NEUTRAL_SIGNAL, NEUTRAL_OUTCOME])

        probs, hidden = network(obs, hidden)
        actions = jax.random.bernoulli(subkey, probs).astype(jnp.int32)

        move_idx = 2 * actions[0] + actions[1]
        new_d  = (d + TURN[move_idx]) % 4
        new_px = (px + DX[d] * MOVE[move_idx]) % PROBE_GRID
        new_py = (py + DY[d] * MOVE[move_idx]) % PROBE_GRID

        new_dist, _, _ = dist_to_mush(new_px, new_py)
        ate = (new_px == mx) & (new_py == my)

        return (new_px, new_py, new_d, hidden, key), (new_dist, ate.astype(jnp.float32))

    hidden = network.init_hidden()
    _, (distances, ates) = jax.lax.scan(step, (px, py, d, hidden, key), None, length=PROBE_STEPS)

    approach_score = (INITIAL_DIST - jnp.min(distances)) / INITIAL_DIST
    ate_any = (ates.sum() > 0).astype(jnp.float32)
    return approach_score, ate_any


@eqx.filter_jit
def run_trial_all_agents(networks, direction, feature_idx, keys):
    """keys: shape (n_agents, n_replicates). Returns arrays of shape
    (n_agents, n_replicates) — each agent's stochastic rollouts for this
    (direction, feature_idx) condition."""
    def per_agent(network, agent_keys):
        return eqx.filter_vmap(lambda k: run_probe_trial(network, direction, feature_idx, k))(agent_keys)

    return eqx.filter_vmap(per_agent)(networks, keys)


def probe_population(networks, key, n_replicates=PROBE_REPLICATES):
    """Run all trials for all agents, averaging n_replicates stochastic
    rollouts per condition. Returns arrays of shape (n_agents,)."""
    n_agents = MAX_AGENTS
    edible_approach  = []
    edible_ate       = []
    poison_approach  = []
    poison_ate       = []

    for feat_idx in range(10, 20):       # edible variants
        for direction in range(4):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, n_agents * n_replicates).reshape(n_agents, n_replicates)
            a, e = run_trial_all_agents(networks, direction, feat_idx, keys)
            edible_approach.append(np.mean(np.array(a), axis=1))
            edible_ate.append(np.mean(np.array(e), axis=1))

    for feat_idx in range(0, 10):        # poisonous variants
        for direction in range(4):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, n_agents * n_replicates).reshape(n_agents, n_replicates)
            a, e = run_trial_all_agents(networks, direction, feat_idx, keys)
            poison_approach.append(np.mean(np.array(a), axis=1))
            poison_ate.append(np.mean(np.array(e), axis=1))

    edible_approach  = np.mean(edible_approach,  axis=0)   # (n_agents,)
    edible_ate       = np.mean(edible_ate,        axis=0)
    poison_approach  = np.mean(poison_approach,  axis=0)
    poison_ate       = np.mean(poison_ate,        axis=0)

    return {
        "edible_approach":  edible_approach,
        "edible_eat_rate":  edible_ate,
        "poison_approach":  poison_approach,
        "poison_eat_rate":  poison_ate,
        "approach_disc":    edible_approach - poison_approach,
        "eat_disc":         edible_ate      - poison_ate,
    }


def load_networks(agents_path, alive_path, h_size=5, recurrent=False):
    """recurrent must match whatever the saved agents were actually trained
    with -- Network and Recurrent_Network have different pytree structures,
    so deserialising against the wrong dummy silently produces garbage
    (or errors) rather than a clean failure."""
    alive = np.load(alive_path).astype(bool)
    dummy_keys = jax.random.split(jax.random.key(0), MAX_AGENTS)
    network_cls = Recurrent_Network if recurrent else Network
    dummy_networks = eqx.filter_vmap(
        lambda k: network_cls(k, input_dim=17, h_size=h_size, output_dim=5)
    )(dummy_keys)
    networks = eqx.tree_deserialise_leaves(agents_path, dummy_networks)
    return networks, alive

    print("Done.")
