from typing import Protocol, Any
import jax
import equinox as eqx


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
