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


class Agents(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    alive: jnp.ndarray
    direction: jnp.ndarray
    last_signal: jnp.ndarray
    network: Network
    hidden: jnp.ndarray
    energy: jnp.ndarray
    mush_cooldown: jnp.ndarray
    last_outcome: jnp.ndarray


class Mushrooms(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    type: jnp.ndarray
    features: jnp.ndarray
    shuffle_countdown: jnp.int32


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
    shuffle_period: int
    frozen_baseline: bool
    no_signal: bool
    recurrent: bool = eqx.field(static=True, default=False)
    pain_pleasure: bool = eqx.field(static=True, default=True)
    h_size: int = eqx.field(static=True, default=5)

    def _build_networks(self, key, n):
        keys = jax.random.split(key, n)
        network_cls = Recurrent_Network if self.recurrent else Network
        network = eqx.filter_vmap(lambda k: network_cls(k, input_dim=17, h_size=self.h_size, output_dim=5))(keys)
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

        # initialise energy and empty mushroom cooldowns
        energy = jnp.where(alive, self.energy_start, 0)
        mush_cooldown = jnp.zeros((max_agents, num_mushroom))
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



        shuffle_countdown = jnp.full((num_mushroom,), self.shuffle_period)


        # create agents and mushrooms data structure
        agents = Agents(posx=posx, posy=posy, alive=alive, direction=direction, last_signal=signal, network=network, hidden=hidden, energy=energy, mush_cooldown=mush_cooldown, last_outcome=last_outcome)
        mushrooms = Mushrooms(posx=mushroom_posx, posy=mushroom_posy, type=mushroom_type, features=mushroom_features, shuffle_countdown=shuffle_countdown)

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

        # mask out mushrooms on cooldown for agents
        cooldown_mask = (agents.mush_cooldown > 0)
        distance_sq = jnp.where(cooldown_mask, jnp.inf, distance_sq)


        # compute directions
        distance = jnp.sqrt(distance_sq + 1e-8)
        inv_dist = 1.0 / distance
        cos_dir = x_diff * inv_dist
        sin_dir = y_diff * inv_dist

        # find the nearest mushroom for each agent
        nearest_mush = jnp.argmin(distance_sq, axis=1)

        # find input direction for nearest mushrooms per agent
        input_cos = cos_dir[jnp.arange(self.max_agents), nearest_mush]
        input_sin = sin_dir[jnp.arange(self.max_agents), nearest_mush]

        # if the agent within perc_radius of nearest mushroom, receive mushroom's perceptual features

        dist_to_mush = distance[jnp.arange(self.max_agents), nearest_mush]
        features = jnp.where(dist_to_mush <= perc_radius, mushrooms.features[nearest_mush], 20)
        features = MUSH_LIBRARY[features]

        # obtain signals produced in last step
        if self.no_signal:
            signals = jnp.full((self.max_agents, 3), SIGNALS[-1])

        else:
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


        obs = jnp.concat([input_cos[:, None], input_sin[:, None], features, signals, agents.last_outcome], axis=1)

        return obs

    def _compute_update(self, key, actions, agents, mushrooms):

        # update agent positions (00 = stay, 10 = turn left, 01 = turn right, 11 = move forward)
        movement = actions[:, :2]

        bit_high = movement[:, 0]
        bit_low = movement[:, 1]

        move_idx = 2 * bit_high + bit_low

        turn = TURN[move_idx]
        move = MOVE[move_idx]

        new_posx = (agents.posx + DX[agents.direction] * move) % self.grid_x
        new_posy = (agents.posy + DY[agents.direction] * move) % self.grid_y
        new_direction = (agents.direction + turn) % 4

        # take agent signal bits and convert back to integer for storage
        signal_bits = actions[:, 2:]
        new_signal = signal_bits[:, 0] * 4 + signal_bits[:, 1] * 2 + signal_bits[:, 2]

        new_posx = jnp.where(agents.alive, new_posx, agents.posx)
        new_posy = jnp.where(agents.alive, new_posy, agents.posy)
        new_direction = jnp.where(agents.alive, new_direction, agents.direction)
        new_signal = jnp.where(agents.alive, new_signal, agents.last_signal)

        # update agents' energy based on mushroom consumption and standard decay
        mush_posx = mushrooms.posx
        mush_posy = mushrooms.posy

        same_x = (new_posx[:, None] == mush_posx[None, :])
        same_y = (new_posy[:, None] == mush_posy[None, :])
        cooldown = agents.mush_cooldown
        edible = (cooldown == 0)

        consume_mushroom = (same_x & same_y & edible & agents.alive[:, None].astype(bool))

        cooldown = jnp.maximum(cooldown - 1, 0)
        cooldown = jnp.where(consume_mushroom, jnp.round((self.mushroom_nutrition/self.energy_decay)+1), cooldown)

        mushroom_type = mushrooms.type

        consume_multiplier = (consume_mushroom * mushroom_type).sum(axis=1)
        energy_change = (consume_multiplier * self.mushroom_nutrition) - self.energy_decay

        energy = jnp.where(agents.alive, agents.energy + energy_change, 0.0)

        alive = jnp.where(energy > 0, 1, 0)

        edible_consumed = (consume_mushroom * (mushroom_type == 1)[None, :]).sum()
        poisonous_consumed = (consume_mushroom * (mushroom_type == self.poison_multiplier)[None, :]).sum()

        # pain/pleasure flash: reflects only this step's consumption, not a running state
        if self.pain_pleasure:
            pleasure = (consume_mushroom & (mushroom_type == 1)[None, :]).any(axis=1)
            pain = (consume_mushroom & (mushroom_type == self.poison_multiplier)[None, :]).any(axis=1)
            last_outcome = jnp.stack([pain, pleasure], axis=1).astype(jnp.float32)
            last_outcome = jnp.where(agents.alive[:, None], last_outcome, 0.0)
        else:
            last_outcome = jnp.zeros_like(agents.last_outcome)

        key, subkey = jax.random.split(key)
        all_cells = jnp.arange(self.grid_x * self.grid_y)
        mush_chosen = jax.random.choice(subkey, all_cells, shape=(self.nb_mushrooms,), replace=False)
        new_mush_posx = mush_chosen // self.grid_y
        new_mush_posy = mush_chosen % self.grid_y
        mush_posx = jnp.where(mushrooms.shuffle_countdown == 0, new_mush_posx, mush_posx)
        mush_posy = jnp.where(mushrooms.shuffle_countdown == 0, new_mush_posy, mush_posy)

        shuffle_countdown = jnp.where(mushrooms.shuffle_countdown == 0, self.shuffle_period,
                                      mushrooms.shuffle_countdown - 1)

        # update agents
        agents = Agents(posx=new_posx, posy=new_posy, alive=alive, direction=new_direction, last_signal=new_signal, network=agents.network, hidden=agents.hidden, energy=energy, mush_cooldown=cooldown, last_outcome=last_outcome)
        mushrooms = Mushrooms(posx=mush_posx, posy=mush_posy, type=mushrooms.type, features=mushrooms.features,
                              shuffle_countdown=shuffle_countdown)


        return agents, mushrooms, edible_consumed, poisonous_consumed

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
        new_cooldown = jnp.where(newborn[:, None], 0, agents.mush_cooldown)

        # newborns start with a blank hidden state rather than inheriting the parent's runtime activations
        new_hidden = jnp.where(newborn.reshape((-1,) + (1,) * (agents.hidden.ndim - 1)), jnp.zeros_like(agents.hidden), agents.hidden)
        new_outcome = jnp.where(newborn[:, None], 0.0, agents.last_outcome)

        became_parent = jnp.zeros(self.max_agents, dtype=bool)
        became_parent = became_parent.at[parent_ranking].set(active_slots)
        new_energy = jnp.where(became_parent, parent_energy, new_energy)

        agents = Agents(posx=new_posx, posy=new_posy, alive=new_alive, direction=new_dir, last_signal=new_signal, network=new_network, hidden=new_hidden, energy=new_energy, mush_cooldown=new_cooldown, last_outcome=new_outcome)

        return agents

    def _compute_respawn(self, key, agents):
        SX = self.grid_x
        SY = self.grid_y
        max_agents = self.max_agents
        num_mushroom = self.nb_mushrooms

        key, subkey = jax.random.split(key)
        all_cells = jnp.arange(SX * SY)
        chosen = jax.random.choice(subkey, all_cells, shape=(max_agents,), replace=False)
        posx = chosen // SY
        posy = chosen % SY

        # initialise agent directions and empty signal
        key, subkey = jax.random.split(key)
        direction = jax.random.randint(subkey, shape=(max_agents,), minval=0, maxval=4)
        signal = jnp.full(max_agents, 8, dtype=jnp.int32)

        # initialise energy and empty mushroom cooldowns
        energy = jnp.full((max_agents,), self.energy_start)
        mush_cooldown = jnp.zeros((max_agents, num_mushroom))
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
        mush_cooldown = jnp.where(dead[:, None], mush_cooldown, agents.mush_cooldown)
        last_outcome = jnp.where(dead[:, None], last_outcome, agents.last_outcome)

        params_new, static_new = eqx.partition(network, eqx.is_inexact_array)
        params_old, _ = eqx.partition(agents.network, eqx.is_inexact_array)
        merged_leaves = jax.tree_util.tree_map(
            lambda n, o: jnp.where(dead.reshape((-1,) + (1,) * (n.ndim - 1)), n, o),
            params_new,params_old)

        new_network = eqx.combine(merged_leaves, static_new)

        new_hidden = jnp.where(dead.reshape((-1,) + (1,) * (hidden.ndim - 1)), hidden, agents.hidden)

        agents = Agents(posx=posx, posy=posy, alive=alive, direction=direction, last_signal=signal, network=new_network, hidden=new_hidden, energy=energy, mush_cooldown=mush_cooldown, last_outcome=last_outcome)

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
        agents, mushrooms, _, _ = self._compute_update(subkey3, actions, agents, mushrooms)

        # compute reproduction or respawn
        if self.frozen_baseline:
            agents = self._compute_respawn(subkey4, agents)
        else:
            mutation_std = self.mutation_std
            agents = self._compute_reproduce(subkey4, agents, mutation_std)


        return (agents, mushrooms)


