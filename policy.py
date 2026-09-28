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

class ApproachMovement(eqx.Module):
    edible_only: bool = False

    def act(self, key, agents, food_states, obs):
        food_layer = obs[..., 0]
        present = (food_layer > 0) if self.edible_only else (food_layer != 0)
        n = present.shape[0]

        ahead = present[:, 1:, 2]
        tier1_hit = ahead.any(axis=-1)

        left_present = present[:, 0, 1]
        right_present = present[:, 0, 3]

        key, tie_key, search_key1, search_key2 = jax.random.split(key, 4)
        tie_break = jax.random.bernoulli(tie_key, p=0.5, shape=(n,))
        flank_turn = jnp.where(left_present & right_present, jnp.where(tie_break, 1, -1),
                                jnp.where(right_present, 1, jnp.where(left_present, -1, 0)))

        tier2_hit = left_present | right_present

        move_roll = jax.random.bernoulli(search_key1, p=0.9, shape=(n,))
        random_turn = jax.random.choice(search_key2, jnp.array([-1, 1]), shape=(n,))

        fallback_turn = jnp.where(move_roll, 0, random_turn)
        fallback_move = jnp.where(move_roll, 1, 0)

        turn = jnp.where(tier1_hit, 0, jnp.where(tier2_hit, flank_turn, fallback_turn))
        move = jnp.where(tier1_hit, 1, jnp.where(tier2_hit, 0, fallback_move))
        return turn, move


class AlwaysEat(eqx.Module):
    def act(self, key, agents, food_states, obs):
        n = agents.posx.shape[0]
        eat_decision = jnp.ones(shape=(n,), dtype=bool)
        return eat_decision

class DiscriminateEat(eqx.Module):
    def act(self, key, agents, food_states, obs):
        food_layer = obs[..., 0]
        own_cell = food_layer[:, 0, 2]
        eat_decision = own_cell > 0
        return eat_decision


wander = ComposedPolicy(movement_rule=RandomMovement(), eat_rule=AlwaysEat())
approach_all = ComposedPolicy(movement_rule=ApproachMovement(edible_only=False), eat_rule=AlwaysEat())
discriminate = ComposedPolicy(movement_rule=ApproachMovement(edible_only=True), eat_rule=DiscriminateEat())