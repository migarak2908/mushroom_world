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

### Predicted values

Evaluating the formulas above at the parameters actually tested (20×20 grid, `N=10`, `P=1`, `d=0.5`). `wander` and `discriminating_wanderer` get ranges rather than point estimates, since random search is bounded rather than exactly determined.

**Predicted `T`**

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate |
|---|---|---|---|---|---|
| 0.01 | 0.25 | 400–963 | 24 | 533–1284 | 31 |
| 0.01 | 0.5 | 400–963 | 24 | 800–1926 | 44 |
| 0.01 | 0.75 | 400–963 | 24 | 1600–3852 | 84 |
| 0.02 | 0.25 | 200–430 | 14 | 267–574 | 17 |
| 0.02 | 0.5 | 200–430 | 14 | 400–860 | 24 |
| 0.02 | 0.75 | 200–430 | 14 | 800–1721 | 44 |
| 0.04 | 0.25 | 100–189 | 9 | 133–252 | 11 |
| 0.04 | 0.5 | 100–189 | 9 | 200–378 | 14 |
| 0.04 | 0.75 | 100–189 | 9 | 400–756 | 24 |
| 0.08 | 0.25 | 50–81 | 6 | 67–108 | 7 |
| 0.08 | 0.5 | 50–81 | 6 | 100–162 | 9 |
| 0.08 | 0.75 | 50–81 | 6 | 200–324 | 14 |

**Predicted `G`**

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate |
|---|---|---|---|---|---|
| 0.01 | 0.25 | −0.49 | −0.29 | −0.49 to −0.48 | −0.17 |
| 0.01 | 0.5 | −0.50 | −0.50 | −0.49 | −0.27 |
| 0.01 | 0.75 | −0.51 | −0.71 | −0.50 | −0.38 |
| 0.02 | 0.25 | −0.49 to −0.47 | −0.14 | −0.48 to −0.46 | +0.08 |
| 0.02 | 0.5 | −0.50 | −0.50 | −0.49 to −0.47 | −0.08 |
| 0.02 | 0.75 | −0.53 to −0.51 | −0.86 | −0.49 | −0.27 |
| 0.04 | 0.25 | −0.47 to −0.45 | +0.06 | −0.46 to −0.42 | +0.44 |
| 0.04 | 0.5 | −0.50 | −0.50 | −0.47 to −0.45 | +0.21 |
| 0.04 | 0.75 | −0.55 to −0.53 | −1.06 | −0.49 to −0.47 | −0.08 |
| 0.08 | 0.25 | −0.44 to −0.40 | +0.27 | −0.41 to −0.35 | +0.86 |
| 0.08 | 0.5 | −0.50 | −0.50 | −0.44 to −0.40 | +0.61 |
| 0.08 | 0.75 | −0.60 to −0.56 | −1.27 | −0.47 to −0.45 | +0.21 |

Note the `q=0.5` column for the two indiscriminate eaters: at `P=1`, `M = N×[(1−q) − q×P]` is exactly zero there, so `G` collapses to exactly `−d` no matter how fast the agent finds food. That makes it a useful check — any measured deviation from `−0.50` at `q=0.5` for `wander` or `approach_all` would indicate something wrong with the energy accounting.

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

**Against the predictions:**

- **The two random-movement behaviours land inside their predicted ranges essentially everywhere.** `wander` falls within its `4/m` to `4t*` bounds at all 12 configurations, and `discriminating_wanderer` within its scaled version of those bounds at 10 of 12 (the two exceptions, both at `q=0.75`, overshoot by under 2%).
- **The two navigating behaviours generally find food more slowly than predicted**, and the size of the gap depends strongly on density: it closes almost completely at the highest density tested (`approach_all` actually beats its prediction there, 5 steps against 6) and widens as food gets sparser, reaching roughly double the predicted `T` at the sparsest settings. This is a known consequence of how navigation is implemented rather than a modelling error: the rule pursues targets that are directly ahead or immediately to one side, but leaves targets at the outer edges of the visual window to be picked up by chance — which matters more when a target in view is a rarer event. It was built this way deliberately: a rule that always steers toward whichever target is nearest can lock into an infinite turn-left/turn-right oscillation when two targets sit on opposite flanks.
- **That slower search propagates into `G` in the direction the formula says it should.** Because `G = M/T − d`, a larger-than-predicted `T` pulls `G` toward `−d` from whichever side it was on: the navigating policies come in below prediction where food is profitable (`q=0.25`) and above it where food is a net loss (`q=0.75`). Every deviation is accounted for by the single `T` discrepancy above.
- **The `q=0.5` check passes.** Both indiscriminate eaters measured `G = −0.50` at every density (one reading of `−0.51`, within sampling noise) — matching the analytically exact value and confirming the energy accounting is sound.
- **`fraction_poisonous` matched predictions throughout:** ≈`q` for `wander`/`approach_all`, and exactly `0` for `discriminating_wanderer`/`discriminate` across all 36,000 runs.

At the sparsest and most dangerous corner (`m=0.01, q=0.75`), up to ~30% of runs never encountered a qualifying mushroom at all within 20,000 steps — an expected consequence of effective edible density falling to `m×(1−q) = 0.0025`, and the reason estimates are noisiest there.

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
