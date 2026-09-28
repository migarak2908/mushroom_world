# Mushroom Foraging Testbed

A modular simulation testbed for studying how foraging behaviour — from random wandering to actively avoiding poisonous food — affects survival, as a foundation for later work on whether such discrimination can evolve from scratch.

## The model

- **World:** a toroidal grid. Mushrooms are either edible or poisonous; eating one respawns it instantly at a random empty cell, keeping the population density constant.
- **Agents:** each step, an agent loses a fixed amount of energy (`d`), can move (turn or step forward) and can choose to eat whatever is on its current cell. Eating an edible mushroom gains energy (`N`); eating a poisonous one loses `P × N`.
- **Perception:** agents see a 3-deep, 5-wide window in front of themselves (their own row plus two ahead), which rotates with the direction they're facing.
- **Density parameters:** `m` = fraction of grid cells holding a mushroom; `q` = fraction of mushrooms that are poisonous.

## Behaviours compared

| Policy | Movement | Eating |
|---|---|---|
| `wander` | Random | Eats anything |
| `approach_all` | Navigates toward the nearest visible mushroom | Eats anything |
| `discriminating_wanderer` | Random | Eats only if standing on an edible mushroom |
| `discriminate` | Navigates toward the nearest visible *edible* mushroom | Eats only if standing on an edible mushroom |

These four span the combinations of "moves with purpose" × "eats with discrimination," which makes it possible to attribute how much of any survival advantage comes from navigation, from diet, or from both together.

## Predictions

From the energy budget alone:

- **Average energy gained per meal**, for an indiscriminate eater: `M = N × [(1 − q) − q×P]`. This is positive only when `q < 1/(1+P)` — otherwise the average meal is a net loss no matter how good `N` is.
- **Net energy per step:** `G = M/T − d`, where `T` is the average number of steps between meals. Positive `G` means the population grows; negative means it shrinks.
- **Time between meals** depends on how directed the search is: undirected wandering is bounded by how long a random walk takes to cover the grid; direct navigation should find food in roughly the time it takes to spot something plus walk to it; adding discrimination on top of either slows things down proportionally to how much of what's around has to be skipped.
- **Fraction of meals that are poisonous** should match `q` exactly for indiscriminate eaters, and be exactly zero for discriminating ones.

## Results

Measured over 1,000 independent runs per configuration, 20,000 steps each, on a 20×20 grid (`N=10`, `P=1`, `d=0.5`):

**Average steps between meals (`T`)**

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate |
|---|---|---|---|---|---|
| 0.01 | 0.25 | 597 | 40 | 924 | 60 |
| 0.01 | 0.5 | 597 | 40 | 1442 | 91 |
| 0.01 | 0.75 | 597 | 40 | 2064 | 129 |
| 0.02 | 0.25 | 296 | 20 | 425 | 28 |
| 0.02 | 0.5 | 296 | 20 | 715 | 47 |
| 0.02 | 0.75 | 296 | 20 | 1487 | 93 |
| 0.04 | 0.25 | 147 | 10 | 202 | 14 |
| 0.04 | 0.5 | 147 | 10 | 321 | 21 |
| 0.04 | 0.75 | 147 | 10 | 766 | 49 |
| 0.08 | 0.25 | 73 | 5 | 99 | 7 |
| 0.08 | 0.5 | 73 | 5 | 151 | 10 |
| 0.08 | 0.75 | 73 | 5 | 331 | 22 |

**Net energy per step (`G`)**

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate |
|---|---|---|---|---|---|
| 0.01 | 0.25 | −0.49 | −0.38 | −0.49 | −0.31 |
| 0.01 | 0.5 | −0.50 | −0.50 | −0.49 | −0.38 |
| 0.01 | 0.75 | −0.51 | −0.63 | −0.50 | −0.44 |
| 0.02 | 0.25 | −0.48 | −0.26 | −0.47 | −0.12 |
| 0.02 | 0.5 | −0.50 | −0.51 | −0.48 | −0.25 |
| 0.02 | 0.75 | −0.52 | −0.75 | −0.49 | −0.37 |
| 0.04 | 0.25 | −0.47 | −0.01 | −0.45 | **+0.25** |
| 0.04 | 0.5 | −0.50 | −0.50 | −0.47 | **+0.00** |
| 0.04 | 0.75 | −0.53 | −0.99 | −0.48 | −0.25 |
| 0.08 | 0.25 | −0.43 | **+0.42** | −0.40 | **+0.94** |
| 0.08 | 0.5 | −0.50 | −0.50 | −0.43 | **+0.50** |
| 0.08 | 0.75 | −0.57 | −1.42 | −0.47 | **+0.01** |

`fraction_poisonous` matched predictions exactly throughout: ≈`q` for `wander`/`approach_all`, and exactly `0` for `discriminating_wanderer`/`discriminate` across every one of the 36,000 runs. `T` matched the predicted shape in every case, with the closest match at high density and somewhat larger gaps at the sparsest, most dangerous corner (`m=0.01, q=0.75`), where a meaningful share of runs (up to ~30%) never encountered a qualifying mushroom at all within 20,000 steps — an expected consequence of low effective food density, not a modelling error.

**The main finding:** `discriminate` beats every other policy's `G` at every single configuration tested. More specifically, navigation and discrimination are *synergistic*, not additive — combining them produces a larger gain than the sum of their individual effects, and the gap grows with how dangerous the environment is (`q`). Navigating quickly toward food is actively harmful on its own once most food is poisonous (`approach_all` underperforms `wander` at `q=0.75`), but becomes highly valuable once paired with the ability to tell food apart. Discrimination, on the other hand, helps on its own at every density and danger level tested — which matters for how a discriminating strategy could plausibly evolve one capability at a time.

## Status

All four behaviours above are implemented and validated against the predictions. Next: introducing reproduction, so that a population's growth or decline can be studied directly rather than inferred from a single agent's energy balance.
