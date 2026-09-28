import jax
import jax.numpy as jnp
import equinox as eqx

from food_source import RespawnStrategy


class Mushrooms(eqx.Module):
    posx: jnp.ndarray
    posy: jnp.ndarray
    type: jnp.ndarray

class MushroomSource(eqx.Module):
    SX: int
    SY: int
    nb_mushrooms: int
    mushroom_nutrition: float
    poison_proportion: float
    poison_multiplier: float
    respawn_strategy: RespawnStrategy

    def reset(self, key):

        key, sk1, sk2 = jax.random.split(key, 3)
        all_cells = jnp.arange(self.SX * self.SY)
        food_cells = jax.random.choice(sk1, all_cells, shape=(self.nb_mushrooms,), replace=False)
        posx, posy = food_cells // self.SY, food_cells % self.SY
        type = jax.random.bernoulli(sk2, p=(1-self.poison_proportion), shape=(self.nb_mushrooms,))
        mushrooms = Mushrooms(posx=posx, posy=posy, type=type)

        return mushrooms

    def consume_and_respawn(self, key, food_state, agent_posx, agent_posy, eat_decision, alive):
        same_x = agent_posx[:, None] == food_state.posx[None, :]
        same_y = agent_posy[:, None] == food_state.posy[None, :]
        eaten_by_agent = same_x & same_y & (eat_decision & alive)[:, None]
        eaten_mask = eaten_by_agent.any(axis=0)

        energy_delta = (eaten_by_agent * jnp.where(food_state.type == 1,
                                                   self.mushroom_nutrition,
                                                   -1 * self.poison_multiplier * self.mushroom_nutrition)).sum(axis=1)

        new_posx, new_posy = self.respawn_strategy.respawn(key, food_state.posx, food_state.posy)

        new_posx = jnp.where(eaten_mask, new_posx, food_state.posx)
        new_posy = jnp.where(eaten_mask, new_posy, food_state.posy)

        mushrooms = Mushrooms(posx=new_posx, posy=new_posy, type=food_state.type)

        return mushrooms, energy_delta

    def type_at(self, food_state, query_posx, query_posy):
        same_x = query_posx[..., None] == food_state.posx
        same_y = query_posy[..., None] == food_state.posy

        match = same_x & same_y

        signed_type = jnp.where(food_state.type, 1, -1)

        return (match * signed_type).sum(axis=-1)












