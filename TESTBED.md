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

**Symbols:** `m` mushroom density, `q` fraction poisonous, `N` energy from an edible mushroom, `P` energy lost from a poisonous one (as a multiple of `N`), `d` energy lost per step, `T` average steps between meals, `G` average net energy change per step.

**Meal value.** An indiscriminate eater gets something edible with probability `(1−q)` and something poisonous with probability `q`, so on average:

```
M = N × [(1 − q) − q×P]
```

This is only positive when `q < 1/(1+P)` — past that point, poison is common or severe enough that eating anything you find is a net loss regardless of `N`. A discriminating eater, which only ever eats edible mushrooms, instead gets `M = N` every meal.

**Net energy per step**, for either kind of eater:

```
G = M/T − d
```

— energy gained per meal, spread over the average wait between meals, minus the constant cost of being alive. Positive `G` means a population would grow; negative means it would shrink.

**Time between meals (`T`)** is where the four behaviours diverge:

- **`wander`** (random movement, eats anything): finding a target covering a fraction `m` of the grid takes at least `4/m` steps — the best case, since only 1 of the 4 possible actions (stay / turn left / turn right / forward) actually reaches a new cell, so covering `1/m` new cells needs roughly `4×(1/m)` steps even with no wasted effort. In practice a random walk revisits ground, so the realistic bound is `4t*`, where `t*` is the grid's random-walk cover time — the `t` solving `π·t / ln(8t) = 1/m`.
- **`approach_all`** (navigates to the nearest visible mushroom, eats anything): `T ≈ 1/(5m) + 4`. Moving forward reveals 5 new cells at a time, so a mushroom enters view roughly every `1/(5m)` steps ("spotting"); reaching it once spotted takes about 4 more steps on average ("reaching").
- **`discriminate`** (navigates to the nearest visible *edible* mushroom, eats only edible): same spotting-and-reaching shape, but only a `(1−q)` fraction of mushrooms are valid targets, so `T ≈ 1/(5m(1−q)) + 4`.
- **`discriminating_wanderer`** (random movement, eats only edible): movement is identical to `wander`, but a poisonous mushroom underfoot doesn't count as a meal — only a `(1−q)` fraction of landings do. So `T ≈ wander's T / (1−q)`.

**Fraction of meals that are poisonous** should match `q` for the indiscriminate eaters, and be exactly zero for the discriminating ones, by construction.

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

## Research plan

The project moves through four stages, each validated before the next is built on top of it.

**Stage 1 — hand-coded baselines (complete, above).** Validate the environment's mechanics and the energy-budget predictions using simple, fixed behaviours, with a single agent and no reproduction — before any learning is involved. This is the foundation everything else depends on: if the environment didn't behave the way the maths predicted here, nothing built on top of it could be trusted either.

**Stage 2 — reproduction, no mutation (next).** A single lineage, starting from one agent, can now reproduce once it accumulates enough energy — testing whether a population sustains or grows itself under a given policy, with the goal of finding parameters where it declines *slowly* rather than exploding or collapsing immediately. Stage 1's results feed directly into this choice:

- `q` has to stay below `1/(1+P)` (`0.5` at `P=1`) — every configuration tested at `q ≥ 0.5` gave `wander` a `G` at or below `−d`, meaning indiscriminate foraging can never sustain a population there no matter how nutrition (`N`) is tuned.
- Even at the one viable `q` (`0.25`), the `N=10` used for Stage 1's validation is far too low — getting `wander` close to a slow decline instead of a sharp one needs `N` roughly 5-50× higher (more at low density, less at high density), since `G` depends on the ratio `N/d`, not either alone.
- The population also has to stay small enough that agents aren't competing with each other for food — see Stage 3.

**Stage 3 — competition.** Many agents sharing one grid at once. Because eaten mushrooms respawn instantly, total food supply never actually drops as the population grows — competition instead shows up as two more specific effects: agents collectively re-searching the grid faster helps undirected wanderers find food sooner, while agents racing each other to a spotted mushroom hurts navigating agents that can lose the race. Both become significant once the population reaches roughly `1/14` to `1/10` of the number of grid cells — a threshold Stage 2 needs to stay under to keep its single-agent-derived parameters valid.

**Stage 4 — evolution.** Replace the hand-coded behaviours with evolved neural controllers (mutation on), to test whether the discrimination advantage measured in Stage 1 actually gets discovered from random starting behaviour — and, since movement and eating are independent outputs, whether it's found by improving navigation first or diet first. The synergy result above (discrimination helps alone at every configuration tested, navigation doesn't) is a concrete prediction for which route evolution is more likely to take, especially in more dangerous environments.
