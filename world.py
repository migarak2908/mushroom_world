import jax
import jax.numpy as jnp
import equinox as eqx


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
    energy: jnp.ndarray



class World(eqx.Module):
    seed: int
    SX: int
    SY: int
    nb_agents: int
    max_agents: int
    energy_start: float
    energy_decay: float
    food_sources: tuple
    death_enabled: bool = eqx.field(static=True, default=True)


    def __post_init__(self):
        if self.energy_decay < 0:
            raise ValueError(f"energy_decay must be positive, got {self.energy_decay}")

    def _reset(self, key):

        SX = self.SX
        SY = self.SY
        nb_agents = self.nb_agents
        max_agents = self.max_agents
        energy_start = self.energy_start

        agent_idx = jnp.arange(max_agents)
        alive = jnp.where(agent_idx < nb_agents, 1, 0)

        key, sk1, sk2 = jax.random.split(key, 3)
        all_cells = jnp.arange(SX * SY)
        chosen = jax.random.choice(sk1, all_cells, shape=(max_agents,), replace=False)
        posx = chosen // SY
        posy = chosen % SY

        direction = jax.random.randint(sk2, shape=(max_agents,), minval=0, maxval=4)
        energy = jnp.where(alive, energy_start, 0)

        agents = Agents(posx=posx,
                        posy=posy,
                        alive=alive,
                        direction=direction,
                        energy=energy)


        food_states = []
        for source in self.food_sources:
            key, food_key = jax.random.split(key)
            food_states.append(source.reset(food_key))

        return agents, tuple(food_states)

    def _step(self, key,  agents, food_states, turn, move, eat_decision):

        agents, food_states = self._apply_eat(key, agents, food_states, eat_decision)
        agents = self._apply_movement(agents, turn, move)
        agents = self._apply_decay(agents)
        agents = self._apply_death(agents)

        return agents, food_states

    def _apply_eat(self, key, agents, food_states, eat_decision):

        energy_delta_total = jnp.zeros_like(agents.energy)
        new_food_states = []
        for source, food_state in zip(self.food_sources, food_states):
            key, food_key = jax.random.split(key)
            food_state, energy_delta = source.consume_and_respawn(
                food_key,
                food_state,
                agents.posx,
                agents.posy,
                eat_decision,
                agents.alive.astype(bool)
            )
            new_food_states.append(food_state)
            energy_delta_total += energy_delta

        agents = eqx.tree_at(lambda a: a.energy, agents, agents.energy + energy_delta_total)

        return agents, tuple(new_food_states)





    def _apply_movement(self,
                       agents,
                       turn,
                       move):

        moving = agents.alive.astype(bool)
        new_posx = jnp.where(moving, (agents.posx + DX[agents.direction] * move) % self.SX, agents.posx)
        new_posy = jnp.where(moving, (agents.posy + DY[agents.direction] * move) % self.SY, agents.posy)
        new_direction = jnp.where(moving, (agents.direction + turn) % 4, agents.direction)

        agents = Agents(posx=new_posx,
                        posy=new_posy,
                        alive=agents.alive,
                        direction=new_direction,
                        energy=agents.energy)

        return agents

    def _apply_decay(self, agents):
        new_energy = jnp.where(agents.alive, agents.energy - self.energy_decay, 0.0)

        agents = Agents(posx=agents.posx,
                        posy=agents.posy,
                        alive=agents.alive,
                        direction=agents.direction,
                        energy=new_energy)

        return agents

    def _apply_death(self, agents):
        if not self.death_enabled:
            return agents

        new_alive = jnp.where(agents.alive.astype(bool) & (agents.energy > 0), 1, 0)

        agents = Agents(posx=agents.posx,
                        posy=agents.posy,
                        alive=new_alive,
                        direction=agents.direction,
                        energy=agents.energy)

        return agents

    def _bearing_and_distance(self,
                              agents,
                              target_posx,
                              target_posy):

        heading_x = DX[agents.direction].astype(jnp.float32)
        heading_y = DY[agents.direction].astype(jnp.float32)

        posx = agents.posx
        posy = agents.posy

        dx = target_posx - posx
        dy = target_posy - posy

        dx = (dx + self.SX // 2) % self.SX - self.SX // 2
        dy = (dy + self.SY // 2) % self.SY - self.SY // 2

        dist = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        cos = heading_x * (dx / dist) + heading_y * (dy / dist)
        sin = heading_x * (dy / dist) - heading_y * (dx / dist)

        return (cos, sin, dist)

    def _window_cells(self, agents):
        fx, fy = DX[agents.direction], DY[agents.direction]
        rx, ry = DY[agents.direction], -DX[agents.direction]

        rows = jnp.array([0, 1, 2])
        cols = jnp.array([-2, -1, 0, 1, 2])

        qx = (agents.posx[:, None, None] + rows[None, :, None]*fx[:, None, None] + cols[None, None, :]*rx[:, None, None]) % self.SX
        qy = (agents.posy[:, None, None] + rows[None, :, None] * fy[:, None, None] + cols[None, None, :] * ry[:, None,
                                                                                   None]) % self.SY

        return qx, qy



def decode_movement(movement_bits):

    bit_high = movement_bits[:, 0]
    bit_low = movement_bits[:, 1]

    move_idx = 2 * bit_high + bit_low

    turn = TURN[move_idx]
    move = MOVE[move_idx]

    return (turn, move)






