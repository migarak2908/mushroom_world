import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Protocol, Any


class FoodSource(Protocol):
    def reset(self, key) -> Any:
        ...

    def consume_and_respawn(self, key, food_state, agent_posx, agent_posy, eat_decision, alive) -> tuple[Any, Any]:
        ...

class RespawnStrategy(Protocol):
    def respawn(self, key, posx, posy) -> tuple[Any, Any]:
        ...

class UniformRespawn(eqx.Module):
    SX: int
    SY: int

    def respawn(self, key, posx, posy):
        all_cells = jnp.arange(self.SX * self.SY)
        occupied_cells = posx * self.SY + posy
        weights = jnp.where(jnp.isin(all_cells, occupied_cells), 0.0, 1.0)

        candidates = jax.random.choice(key, all_cells, shape=posx.shape, replace=False, p=weights)

        new_posx, new_posy = candidates // self.SY, candidates % self.SY

        return new_posx, new_posy

