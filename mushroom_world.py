from dataclasses import dataclass
from typing import ClassVar, Optional

import jax
import jax.numpy as jnp
import equinox as eqx


from agent import Network, Recurrent_Network

# Initialise a world that is a grid of size X, Y. Populate this grid with n agents at random locations.
# Populate this grid with mushrooms at random locations. Edible mushrooms should cover 2.5% of the total
# grid area. Each agent should be comprised of a neural network (basic - MLP; advanced MLP + LSTM).
# Each agent receives an input of 8 rays (for each direction) with max distance and features of intervening
# objects (16 features x 8 rays = input vector). They have an output of 7 (3 for signal and 4 for movement).

# Create array of valid signals
n = 3
bits = jnp.arange(2**n)
SIGNALS = (bits[:, None] >> jnp.arange(n)[::-1]) & 1
SIGNALS = jnp.concat([SIGNALS, jnp.array([[0.5, 0.5, 0.5]])])


# Create array of valid mushroom features
mushroom_prototype = jnp.array([
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
])

bit_change = jnp.eye(10, dtype=jnp.int32)
MUSH_LIBRARY = mushroom_prototype[:, None, :] ^ bit_change[None, :, :]
MUSH_LIBRARY = jnp.reshape(MUSH_LIBRARY, (-1, 10))
MUSH_LIBRARY = jnp.concat([MUSH_LIBRARY, jnp.full((1, 10), 0.5)])

# Position updates based on direction
TURN = jnp.array([0, 1, -1, 0])
MOVE = jnp.array([0, 0, 0, 1])
DX = jnp.array([0, 1, 0, -1])
DY = jnp.array([1, 0, -1, 0])


@dataclass(frozen=True)
class ObsLayout:
    """Index layout of the observation vector, derived from a world's
    signalling / consequence_inputs flags. Base 13 dims (always present):
    bearing to the nearest mushroom in the agent's own egocentric frame
    (cos dtheta, sin dtheta -- 0 deg = straight ahead, see _compute_obs),
    normalised distance to it, and its 10 feature bits. The signal (3) and
    outcome/pain-pleasure (2) blocks are appended, in that order, only when
    their flag is on. Consumers should read fields by name, never by raw
    position.
    """
    signalling: bool
    consequence_inputs: bool

    cos: ClassVar[int] = 0
    sin: ClassVar[int] = 1
    dist: ClassVar[int] = 2
    features: ClassVar[slice] = slice(3, 13)

    @property
    def signal(self) -> Optional[slice]:
        return slice(13, 16) if self.signalling else None

    @property
    def outcome(self) -> Optional[slice]:
        if not self.consequence_inputs:
            return None
        base = 16 if self.signalling else 13
        return slice(base, base + 2)

    @property
    def dim(self) -> int:
        d = 13
        if self.signalling:
            d += 3
        if self.consequence_inputs:
            d += 2
        return d


@dataclass(frozen=True)
class ActionLayout:
    """Index layout of the action vector: movement (2 bits) and eat (1 bit)
    are always present; the 3 signal bits are appended only when signalling
    is on.
    """
    signalling: bool

    move_hi: ClassVar[int] = 0
    move_lo: ClassVar[int] = 1
    eat: ClassVar[int] = 2

    @property
    def signal(self) -> Optional[slice]:
        return slice(3, 6) if self.signalling else None

    @property
    def dim(self) -> int:
        return 6 if self.signalling else 3


class Agents(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    alive: jnp.ndarray
    direction: jnp.ndarray
    last_signal: jnp.ndarray
    network: Network
    hidden: jnp.ndarray
    energy: jnp.ndarray
    last_outcome: jnp.ndarray


class Mushrooms(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    type: jnp.ndarray
    features: jnp.ndarray
    regrowth_countdown: jnp.ndarray


class MushroomWorld(eqx.Module):
    seed: int
    grid_x: int
    grid_y: int
    nb_agents: int
    max_agents: int
    nb_mushrooms: int
    energy_start: float
    energy_decay: float
    mushroom_nutrition: float
    poison_multiplier: int
    reprod_threshold: float
    reprod_cost: float
    mutation_std: float
    regrowth_period: int
    frozen_baseline: bool
    signalling: bool = eqx.field(static=True, default=True)
    consequence_inputs: bool = eqx.field(static=True, default=True)
    allow_death: bool = eqx.field(default=True)
    allow_reproduction: bool = eqx.field(default=True)
    recurrent: bool = eqx.field(static=True, default=False)
    h_size: int = eqx.field(static=True, default=5)

    @property
    def obs_layout(self) -> ObsLayout:
        return ObsLayout(signalling=self.signalling, consequence_inputs=self.consequence_inputs)

    @property
    def action_layout(self) -> ActionLayout:
        return ActionLayout(signalling=self.signalling)

    def _build_networks(self, key, n):
        keys = jax.random.split(key, n)
        network_cls = Recurrent_Network if self.recurrent else Network
        network = eqx.filter_vmap(lambda k: network_cls(
            k, input_dim=self.obs_layout.dim, h_size=self.h_size, output_dim=self.action_layout.dim
        ))(keys)
        hidden = eqx.filter_vmap(lambda n: n.init_hidden())(network)
        return network, hidden

    def reset_fn(self):

        SX = self.grid_x
        SY = self.grid_y
        nb_agents = self.nb_agents
        max_agents = self.max_agents
        num_mushroom = self.nb_mushrooms

        # set key
        key = jax.random.key(self.seed)

        # initialise live agents
        agent_idx = jnp.arange(max_agents)
        alive = jnp.where(agent_idx < nb_agents, 1, 0)

        # initialise agent positions
        key, subkey = jax.random.split(key)
        all_cells = jnp.arange(SX * SY)
        chosen = jax.random.choice(subkey, all_cells, shape=(max_agents,), replace=False)
        posx = chosen // SY
        posy = chosen % SY

        # initialise agent directions and empty signal
        key, subkey = jax.random.split(key)
        direction = jax.random.randint(subkey, shape=(max_agents,), minval=0, maxval=4)
        signal = jnp.full(max_agents, 8, dtype=jnp.int32)

        # initialise energy and outcome buffers
        energy = jnp.where(alive, self.energy_start, 0)
        last_outcome = jnp.zeros((max_agents, 2))

        # initialise agent params
        key, subkey = jax.random.split(key)
        network, hidden = self._build_networks(subkey, max_agents)

        # initialise mushroom positions, type and features

        key, subkey = jax.random.split(key)
        mush_chosen = jax.random.choice(subkey, all_cells, shape=(num_mushroom,), replace=False)
        mushroom_posx = mush_chosen // SY
        mushroom_posy = mush_chosen % SY

        mushroom_type = jnp.where(jnp.arange(num_mushroom) < num_mushroom // 2, self.poison_multiplier, 1)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        mushroom_features = jnp.where(mushroom_type==1,
                                      jax.random.randint(subkey1, shape=(num_mushroom,), minval=10, maxval=20),
                                      jax.random.randint(subkey2, shape=(num_mushroom,), minval=0, maxval=10))



        regrowth_countdown = jnp.zeros((num_mushroom,), dtype=jnp.int32)


        # create agents and mushrooms data structure
        agents = Agents(posx=posx, posy=posy, alive=alive, direction=direction, last_signal=signal, network=network, hidden=hidden, energy=energy, last_outcome=last_outcome)
        mushrooms = Mushrooms(posx=mushroom_posx, posy=mushroom_posy, type=mushroom_type, features=mushroom_features, regrowth_countdown=regrowth_countdown)

        return (agents, mushrooms)


    def _compute_obs(self, key, agents, mushrooms, perc_radius):
        key, subkey = jax.random.split(key)

        SX = self.grid_x
        SY = self.grid_y

        # compute distance matrix
        x_diff = agents.posx[:, None] - mushrooms.posx[None, :]
        y_diff = agents.posy[:, None] - mushrooms.posy[None, :]
        x_diff = (x_diff + SX // 2) % SX - SX // 2
        y_diff = (y_diff + SY // 2) % SY - SY // 2

        distance_sq = x_diff ** 2 + y_diff ** 2
        distance_sq = distance_sq + jax.random.uniform(subkey, distance_sq.shape) * 1e-6

        # mask out mushrooms that are currently regrowing (absent for everyone)
        absent = (mushrooms.regrowth_countdown > 0)
        distance_sq = jnp.where(absent[None, :], jnp.inf, distance_sq)


        # bearing FROM the agent TO each mushroom, world-frame (x_diff/y_diff
        # above are agent-minus-mushroom, so negate)
        distance = jnp.sqrt(distance_sq + 1e-8)
        inv_dist = 1.0 / distance
        world_cos = -x_diff * inv_dist
        world_sin = -y_diff * inv_dist

        # find the nearest mushroom for each agent
        nearest_mush = jnp.argmin(distance_sq, axis=1)

        world_cos_nearest = world_cos[jnp.arange(self.max_agents), nearest_mush]
        world_sin_nearest = world_sin[jnp.arange(self.max_agents), nearest_mush]

        # rotate into the agent's own egocentric frame (0 deg = straight
        # ahead) via dot/cross with its heading unit vector. Raw world-frame
        # bearing does not change when an agent turns in place -- an agent
        # only ever observes its own heading through this rotation, since
        # direction is otherwise never exposed in the observation.
        heading_x = DX[agents.direction].astype(jnp.float32)
        heading_y = DY[agents.direction].astype(jnp.float32)
        input_cos = heading_x * world_cos_nearest + heading_y * world_sin_nearest
        input_sin = heading_x * world_sin_nearest - heading_y * world_cos_nearest

        # if the agent within perc_radius of nearest mushroom, receive mushroom's perceptual features

        dist_to_mush = distance[jnp.arange(self.max_agents), nearest_mush]
        features = jnp.where(dist_to_mush <= perc_radius, mushrooms.features[nearest_mush], 20)
        features = MUSH_LIBRARY[features]

        # normalised distance to nearest mushroom: 0 = on top of it, 1 = at or
        # beyond perc_radius (matches the range over which features are visible)
        input_dist = jnp.clip(dist_to_mush / perc_radius, 0.0, 1.0)

        parts = [input_cos[:, None], input_sin[:, None], input_dist[:, None], features]

        # obtain signals produced in last step
        if self.signalling:
            last_signal = agents.last_signal

            # find the 2 agents closest to each mushroom
            closest_agents = jnp.argsort(distance, axis=0)[:2]
            nearest_agent_per_mush = closest_agents[0, :]
            backup_agent_per_mush = closest_agents[1, :]

            # for each agent find the agent closest to its nearest mushroom and second closest
            agent_at_mush = nearest_agent_per_mush[nearest_mush]
            backup_at_mush = backup_agent_per_mush[nearest_mush]

            # for each agent if the agent closest to its nearest mushroom is itself, use the second-nearest
            agents_idx = jnp.arange(self.max_agents)
            use_backup = (agents_idx == agent_at_mush)

            signalling_agents = jnp.where(use_backup, backup_at_mush, agent_at_mush)

            # only take signals from agents within the perception radius of the nearest mushroom
            signalling_dist = distance[signalling_agents, nearest_mush]
            signalling_alive = agents.alive[signalling_agents]
            signals = jnp.where((signalling_dist <= perc_radius) & signalling_alive, last_signal[signalling_agents], 8)

            signals = SIGNALS[signals]
            parts.append(signals)

        if self.consequence_inputs:
            parts.append(agents.last_outcome)

        obs = jnp.concat(parts, axis=1)

        return obs

    def _compute_update(self, key, actions, agents, mushrooms):

        layout = self.action_layout

        # decode network outputs: movement (2) + eat (1) [+ signal (3)]
        movement = actions[:, :2]
        eat_action = actions[:, layout.eat].astype(bool)

        # --- resolve eating on the current cell, before movement ---
        # a mushroom is edible iff present (not regrowing); an agent consumes it
        # only if it is standing on the cell and chooses to eat. a contested
        # mushroom goes to one uniformly-random eater; the rest waste the action.
        mush_present = (mushrooms.regrowth_countdown == 0)

        same_x = (agents.posx[:, None] == mushrooms.posx[None, :])
        same_y = (agents.posy[:, None] == mushrooms.posy[None, :])
        wants = (same_x & same_y & mush_present[None, :]
                 & agents.alive[:, None].astype(bool) & eat_action[:, None])

        key, sk_tie = jax.random.split(key)
        priority = jax.random.uniform(sk_tie, (self.max_agents,))
        winner = jnp.argmax(jnp.where(wants, priority[:, None], -jnp.inf), axis=0)
        mush_has_eater = wants.any(axis=0)
        consume_mushroom = jnp.zeros_like(wants).at[
            winner, jnp.arange(self.nb_mushrooms)
        ].set(mush_has_eater)

        mushroom_type = mushrooms.type
        consume_multiplier = (consume_mushroom * mushroom_type).sum(axis=1)
        energy_change = (consume_multiplier * self.mushroom_nutrition) - self.energy_decay

        energy = jnp.where(agents.alive, agents.energy + energy_change, 0.0)
        if self.allow_death:
            alive = jnp.where(energy > 0, 1, 0)
        else:
            alive = agents.alive

        edible_consumed = (consume_mushroom * (mushroom_type == 1)[None, :]).sum()
        poisonous_consumed = (consume_mushroom * (mushroom_type == self.poison_multiplier)[None, :]).sum()

        # pain/pleasure flash: reflects only this step's consumption, not a running state
        if self.consequence_inputs:
            pleasure = (consume_mushroom & (mushroom_type == 1)[None, :]).any(axis=1)
            pain = (consume_mushroom & (mushroom_type == self.poison_multiplier)[None, :]).any(axis=1)
            last_outcome = jnp.stack([pain, pleasure], axis=1).astype(jnp.float32)
            last_outcome = jnp.where(alive[:, None], last_outcome, 0.0)
        else:
            last_outcome = jnp.zeros_like(agents.last_outcome)

        # --- movement (agents that died this step do not move) ---
        move_idx = 2 * movement[:, 0] + movement[:, 1]
        turn = TURN[move_idx]
        move = MOVE[move_idx]
        moving = alive.astype(bool)

        new_posx = jnp.where(moving, (agents.posx + DX[agents.direction] * move) % self.grid_x, agents.posx)
        new_posy = jnp.where(moving, (agents.posy + DY[agents.direction] * move) % self.grid_y, agents.posy)
        new_direction = jnp.where(moving, (agents.direction + turn) % 4, agents.direction)

        if self.signalling:
            signal_bits = actions[:, layout.signal]
            new_signal = signal_bits[:, 0] * 4 + signal_bits[:, 1] * 2 + signal_bits[:, 2]
            new_signal = jnp.where(moving, new_signal, agents.last_signal)
        else:
            new_signal = agents.last_signal

        # --- mushroom regrowth: relocate at the moment of eating (position is
        # unobservable while absent anyway, so this is correct even for
        # regrowth_period == 0, i.e. instant regrowth) to a fresh cell not
        # currently occupied by any other visible mushroom; becomes visible
        # again after regrowth_period steps. type is kept, features are
        # freshly resampled within the same class. Assumes free cells >=
        # nb_mushrooms.
        #
        # The relocation search (occupied-cell mask + argsort over every grid
        # cell) is the most expensive part of a step and is only ever needed
        # on the rare steps where something was actually eaten -- lax.cond
        # skips it entirely otherwise, which matters a lot for a lone agent
        # (eats roughly once every several-to-many steps) over long scans.
        old_countdown = mushrooms.regrowth_countdown
        mushroom_eaten = consume_mushroom.any(axis=0)
        regrowth_countdown = jnp.where(
            mushroom_eaten, self.regrowth_period, jnp.maximum(old_countdown - 1, 0),
        )

        def _relocate(mushrooms, mushroom_eaten, mush_present, key):
            still_present = mush_present & ~mushroom_eaten
            flat_idx = mushrooms.posx * self.grid_y + mushrooms.posy
            occupied_cell = jnp.zeros(self.grid_x * self.grid_y, dtype=bool).at[flat_idx].max(still_present)

            sk_cell, sk_feat1, sk_feat2 = jax.random.split(key, 3)
            priority = jnp.where(occupied_cell, -jnp.inf, jax.random.uniform(sk_cell, (self.grid_x * self.grid_y,)))
            free_cells = jnp.argsort(-priority)[: self.nb_mushrooms]

            serial_rank = jnp.clip(jnp.cumsum(mushroom_eaten.astype(jnp.int32)) - 1, 0, self.nb_mushrooms - 1)
            new_cell = free_cells[serial_rank]
            new_posx = jnp.where(mushroom_eaten, new_cell // self.grid_y, mushrooms.posx)
            new_posy = jnp.where(mushroom_eaten, new_cell % self.grid_y, mushrooms.posy)

            new_feat = jnp.where(
                mushrooms.type == 1,
                jax.random.randint(sk_feat1, (self.nb_mushrooms,), minval=10, maxval=20),
                jax.random.randint(sk_feat2, (self.nb_mushrooms,), minval=0, maxval=10),
            )
            new_features = jnp.where(mushroom_eaten, new_feat, mushrooms.features)
            return new_posx, new_posy, new_features

        def _no_relocate(mushrooms, mushroom_eaten, mush_present, key):
            return mushrooms.posx, mushrooms.posy, mushrooms.features

        key, sk_relocate = jax.random.split(key)
        mush_posx, mush_posy, mush_features = jax.lax.cond(
            jnp.any(mushroom_eaten), _relocate, _no_relocate,
            mushrooms, mushroom_eaten, mush_present, sk_relocate,
        )

        agents = Agents(posx=new_posx, posy=new_posy, alive=alive, direction=new_direction,
                        last_signal=new_signal, network=agents.network, hidden=agents.hidden,
                        energy=energy, last_outcome=last_outcome)
        mushrooms = Mushrooms(posx=mush_posx, posy=mush_posy, type=mushrooms.type,
                              features=mush_features, regrowth_countdown=regrowth_countdown)

        # per-agent: the type-multiplier of whatever mushroom it ate this step
        # (0 = no meal, 1 = edible, poison_multiplier = poison); each agent
        # eats at most one mushroom per step under the tie-break above.
        return agents, mushrooms, edible_consumed, poisonous_consumed, consume_multiplier

    def _compute_reproduce(self, key, agents, mutation_std):

        # identify reproducers and empty slots for newborns to take
        reproducers = (agents.energy > self.reprod_threshold)
        dead_slots = (agents.alive == 0)

        # rank reproducers by energy i.e. highest energy reproducers have priority
        parents_energy = jnp.where(reproducers, agents.energy, -jnp.inf)
        parent_ranking = jnp.argsort(-parents_energy)

        # send empty slot indices to front (to align with parent ranking)
        dead_slots_sorted = jnp.argsort(~dead_slots)

        # find the minimum out of the number of empty slots and the number of viable reproducers
        n_pairs = jnp.minimum(reproducers.sum(), dead_slots.sum())
        active_slots = (jnp.arange(self.max_agents) < n_pairs)

        # determine which slots will take on a newborn and identify the parent idx for that newborn
        newborn = jnp.zeros(self.max_agents, dtype=bool)
        newborn = newborn.at[dead_slots_sorted].set(active_slots)

        parent_idx = jnp.zeros(self.max_agents, dtype=jnp.int32)
        parent_idx = parent_idx.at[dead_slots_sorted].set(parent_ranking)

        # partition the array (e.g. weights and bias) and non-array (e.g. activation funcs.) of the networks for mutation
        params, static = eqx.partition(agents.network, eqx.is_inexact_array)

        # flatten networks' pytree structures and prepare keys
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(leaves)+1)
        mutation_keys, dir_key = keys[:-1], keys[-1]

        # function takes key and network array batches and applies mutation, where newborn
        def mutate_leaves(mutation_key, leaf):
            parent_values = leaf[parent_idx]
            noise = mutation_std * jax.random.normal(mutation_key, parent_values.shape)
            newborn_values = parent_values + noise
            newborn_mask = newborn.reshape((-1,) + (1,) * (leaf.ndim - 1))

            return jnp.where(newborn_mask, newborn_values, leaf)

        # apply function and rebuild trees to obtain new networks for newborns
        new_leaves = [mutate_leaves(k, leaf) for k, leaf in zip(mutation_keys, leaves)]
        new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
        new_network = eqx.combine(new_params, static)

        # determine new attributes
        child_posx = agents.posx[parent_idx]
        child_posy = agents.posy[parent_idx]
        child_dir = jax.random.randint(dir_key, (self.max_agents,), minval=0, maxval=4)

        parent_energy = agents.energy - self.reprod_cost
        child_energy = self.energy_start

        new_posx = jnp.where(newborn, child_posx, agents.posx)
        new_posy = jnp.where(newborn, child_posy, agents.posy)
        new_alive = jnp.where(newborn, 1, agents.alive)
        new_dir = jnp.where(newborn, child_dir, agents.direction)
        new_signal = jnp.where(newborn, 8, agents.last_signal)
        new_energy = jnp.where(newborn, child_energy, agents.energy)

        # newborns start with a blank hidden state rather than inheriting the parent's runtime activations
        new_hidden = jnp.where(newborn.reshape((-1,) + (1,) * (agents.hidden.ndim - 1)), jnp.zeros_like(agents.hidden), agents.hidden)
        new_outcome = jnp.where(newborn[:, None], 0.0, agents.last_outcome)

        became_parent = jnp.zeros(self.max_agents, dtype=bool)
        became_parent = became_parent.at[parent_ranking].set(active_slots)
        new_energy = jnp.where(became_parent, parent_energy, new_energy)

        agents = Agents(posx=new_posx, posy=new_posy, alive=new_alive, direction=new_dir, last_signal=new_signal, network=new_network, hidden=new_hidden, energy=new_energy, last_outcome=new_outcome)

        return agents

    def _compute_respawn(self, key, agents):
        SX = self.grid_x
        SY = self.grid_y
        max_agents = self.max_agents

        key, subkey = jax.random.split(key)
        all_cells = jnp.arange(SX * SY)
        chosen = jax.random.choice(subkey, all_cells, shape=(max_agents,), replace=False)
        posx = chosen // SY
        posy = chosen % SY

        # initialise agent directions and empty signal
        key, subkey = jax.random.split(key)
        direction = jax.random.randint(subkey, shape=(max_agents,), minval=0, maxval=4)
        signal = jnp.full(max_agents, 8, dtype=jnp.int32)

        # initialise energy and outcome buffers
        energy = jnp.full((max_agents,), self.energy_start)
        last_outcome = jnp.zeros((max_agents, 2))

        # initialise agent params
        key, subkey = jax.random.split(key)
        network, hidden = self._build_networks(subkey, max_agents)

        dead = (agents.alive == 0)
        posx = jnp.where(dead, posx, agents.posx)
        posy = jnp.where(dead, posy, agents.posy)
        alive = jnp.where(dead, 1, agents.alive)
        direction = jnp.where(dead, direction, agents.direction)
        signal = jnp.where(dead, signal, agents.last_signal)
        energy = jnp.where(dead, energy, agents.energy)
        last_outcome = jnp.where(dead[:, None], last_outcome, agents.last_outcome)

        params_new, static_new = eqx.partition(network, eqx.is_inexact_array)
        params_old, _ = eqx.partition(agents.network, eqx.is_inexact_array)
        merged_leaves = jax.tree_util.tree_map(
            lambda n, o: jnp.where(dead.reshape((-1,) + (1,) * (n.ndim - 1)), n, o),
            params_new,params_old)

        new_network = eqx.combine(merged_leaves, static_new)

        new_hidden = jnp.where(dead.reshape((-1,) + (1,) * (hidden.ndim - 1)), hidden, agents.hidden)

        agents = Agents(posx=posx, posy=posy, alive=alive, direction=direction, last_signal=signal, network=new_network, hidden=new_hidden, energy=energy, last_outcome=last_outcome)

        return agents

    def step_fn(self, key, agents, mushrooms, perc_radius):

        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)

        # obtain observations
        obs = self._compute_obs(subkey1, agents, mushrooms, perc_radius)

        # compute agent actions given observations
        probs, hidden = eqx.filter_vmap(lambda n, o, h: n(o, h))(agents.network, obs, agents.hidden)
        actions = jax.random.bernoulli(subkey2, probs).astype(jnp.int32)
        agents = eqx.tree_at(lambda a: a.hidden, agents, hidden)

        # update agents and mushrooms given agent actions
        agents, mushrooms, _, _, _ = self._compute_update(subkey3, actions, agents, mushrooms)

        # compute reproduction or respawn
        if self.allow_reproduction:
            if self.frozen_baseline:
                agents = self._compute_respawn(subkey4, agents)
            else:
                agents = self._compute_reproduce(subkey4, agents, self.mutation_std)

        return (agents, mushrooms)

    def step_fn_policy(self, key, agents, mushrooms, perc_radius, policy, policy_state):
        """Drive the population with a hand-coded policy instead of the
        neural network -- agents.network/hidden are untouched. `policy` is a
        pure function `(obs_row, key, state_row) -> (action_row, new_state_row)`,
        applied per-agent under vmap; obs/action layouts follow
        self.obs_layout / self.action_layout. `policy_state` is an
        arbitrary per-agent pytree the policy may use to carry memory across
        steps (e.g. controllers.make_discriminate's escaping/steps_left);
        stateless policies just pass it through unchanged (see
        controllers.epsilon_greedy). Returns edible/poisonous consumption
        counts (unlike step_fn) and the updated policy_state, since
        calibration needs both every step.
        """
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)

        obs = self._compute_obs(subkey1, agents, mushrooms, perc_radius)
        policy_keys = jax.random.split(subkey2, self.max_agents)
        actions, policy_state = jax.vmap(policy)(obs, policy_keys, policy_state)

        agents, mushrooms, edible_consumed, poisonous_consumed, consume_multiplier = self._compute_update(
            subkey3, actions, agents, mushrooms
        )

        if self.allow_reproduction:
            if self.frozen_baseline:
                agents = self._compute_respawn(subkey4, agents)
            else:
                agents = self._compute_reproduce(subkey4, agents, self.mutation_std)

        return agents, mushrooms, edible_consumed, poisonous_consumed, consume_multiplier, policy_state
