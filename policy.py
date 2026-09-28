from typing import Protocol, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from world import decode_movement

class Policy(Protocol):
    def act(self, key, agents, food_states, obs) -> tuple[Any, Any, Any]:
        ...

class MovementRule(Protocol):
    def act(self, key, agents, food_states, obs) -> tuple[Any, Any]:
        ...

class EatRule(Protocol):
    def act(self, key, agents, food_states, obs) -> Any:
        ...


class ComposedPolicy(eqx.Module):
    movement_rule: MovementRule
    eat_rule: EatRule

    def act(self, key, agents, food_states, obs):
        key, move_key, eat_key = jax.random.split(key, 3)
        turn, move = self.movement_rule.act(move_key, agents, food_states, obs)
        eat_decision = self.eat_rule.act(eat_key, agents, food_states, obs)

        return turn, move, eat_decision


class RandomMovement(eqx.Module):
    def act(self, key, agents, food_states, obs):
        n = agents.posx.shape[0]
        movement_bits = jax.random.bernoulli(key, p=0.5, shape=(n, 2)).astype(jnp.int32)
        turn, move = decode_movement(movement_bits)
        return turn, move

class AlwaysEat(eqx.Module):
    def act(self, key, agents, food_states, obs):
        n = agents.posx.shape[0]
        eat_decision = jnp.ones(shape=(n,), dtype=bool)
        return eat_decision

