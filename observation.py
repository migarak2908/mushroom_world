import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Protocol, Any


class ObservationChannel(Protocol):
    def compute(self, agents, food_states, qx, qy) -> jnp.ndarray:
        ...

class FoodTypeChannel(eqx.Module):
    source: Any
    source_idx: int

    def compute(self, agents, food_states, qx, qy):
        food_state = food_states[self.source_idx]
        food_obs = self.source.type_at(food_state, qx, qy)

        return food_obs