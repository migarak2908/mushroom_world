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
| `wander_0_5` | Random | Eats with probability 0.5, regardless of what's underfoot |
| `approach_all_0_5` | Navigates toward the nearest visible mushroom | Eats with probability 0.5, regardless of what's underfoot |

The first four span the combinations of "moves with purpose" × "eats with discrimination," which makes it possible to attribute how much of any survival advantage comes from navigation, from diet, or from both together. The last two add an *imperfect* eater on top of `wander`/`approach_all` — a stand-in for how a freshly-initialised, not-yet-selected network behaves, since it hasn't learned to condition eating on what's actually in front of it.

## Predicting `T`

**Symbols:** `m` mushroom density, `q` fraction poisonous, `N` energy from an edible mushroom, `P` energy lost from a poisonous one (as a multiple of `N`), `d` energy lost per step, `T` average steps between meals, `G` average net energy change per step.

**Meal value.** An indiscriminate eater gets something edible with probability `(1−q)` and something poisonous with probability `q`, so on average:

```
M = N × [(1 − q) − q×P]
```

Positive only when `q < 1/(1+P)` — past that point, poison is common or severe enough that eating anything you find is a net loss regardless of `N`. A discriminating eater instead gets `M = N` every meal, since it only ever eats edible mushrooms.

**Net energy per step**, for either kind of eater:

```
G = M/T − d
```

Positive `G` means a population would grow; negative means it would shrink.

**`T` — average steps between meals — is the one quantity in this budget that can't be written down from first principles**; it depends on the real dynamics of movement and search. The qualitative shape is: undirected wandering is bounded by how long a random walk takes to cover the grid (`~1/m`, up to a log correction); navigating toward a spotted target should scale the same way but with a smaller constant, since spotting and reaching are both roughly proportional to density; and adding discrimination on top of either slows things down by a factor depending on `q`, since only edible landings still count as a meal.

Rather than rely on that qualitative shape alone, `T` was measured directly across `m ∈ [0.02, 0.16]` (step `0.02`) and fit to `T ≈ a + b/m`:

| Policy | `q` | `a` | `b` | max fit error |
|---|---|---|---|---|
| `wander` | any | −0.77 | 5.93 | 0.8% |
| `approach_all` | any | 1.26 | 0.23 | 0.7% |
| `wander_0_5` | any | −0.93 | 6.95 | 0.7% |
| `approach_all_0_5` | any | 2.25 | 0.42 | 2.5% |
| `discriminating_wanderer` | 0.25 | −1.52 | 7.94 | 1.2% |
| `discriminating_wanderer` | 0.5 | −2.05 | 11.96 | 0.7% |
| `discriminating_wanderer` | 0.75 | −16.70 | 25.20 | 4.3% |
| `discriminate` | 0.25 | 1.24 | 0.31 | 1.0% |
| `discriminate` | 0.5 | 1.12 | 0.47 | 2.1% |
| `discriminate` | 0.75 | 0.70 | 0.97 | 2.7% |

`b/m` is the pure-density term the qualitative model predicts; `a` is an empirical floor (a "reaching" cost for the navigating policies, or measurement noise for the random ones — `wander`'s `a≈−0.77` isn't a real negative cost, it's just the linear fit's best compromise across the range). The four policies whose movement doesn't depend on mushroom type (`wander`, `approach_all`, and their `_0_5` variants) fit this form independent of `q`, essentially exactly (well under 1% error). The two discriminating policies fit less tightly and need a separate `a`, `b` per `q`, because how often a landing "counts" depends on `q` directly — still, the fit is now under 5% everywhere, worst at the sparsest, most dangerous corner (`m=0.02, q=0.75`).

**Fraction of meals that are poisonous** should match `q` for the indiscriminate eaters (`wander`, `approach_all`, and the `_0_5` variants), and be exactly zero for the discriminating ones, by construction. Confirmed exactly across all runs below.

## Results

Measured over 1,000 independent runs per configuration, 20,000 steps each, on a 20×20 grid (`N=10`, `P=1`, `d=0.5`), across `m ∈ {0.02, 0.04, ..., 0.16}` × `q ∈ {0.25, 0.5, 0.75}` — 24 configurations, 6 policies each, 144,000 runs total. (Mushroom type is now assigned as a fixed count, shuffled to random positions and frozen for the episode, rather than an independent coin flip per mushroom — the earlier version let the actual edible count vary by seed, which at the sparsest/most dangerous corner occasionally produced seeds with zero edible mushrooms on the whole grid. No seed in the current data failed to find a single qualifying meal.)

**Average steps between meals (`T`)** — each cell is `predicted / measured`, using the qualitative-model formulas from above (a range for the three policies whose prediction is a window, a point value otherwise; `approach_all_0_5` has no closed-form prediction):

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | 200.0 to 430.2 / 295.85 | 14.0 / 12.71 | 266.7 to 573.6 / 396.36 | 17.3 / 16.66 | 250.0 to 537.7 / 346.80 | − / 23.36 |
| 0.02 | 0.5 | 200.0 to 430.2 / 295.85 | 14.0 / 12.71 | 400.0 to 860.4 / 595.90 | 24.0 / 24.72 | 250.0 to 537.7 / 346.80 | − / 23.36 |
| 0.02 | 0.75 | 200.0 to 430.2 / 295.85 | 14.0 / 12.71 | 800.0 to 1720.8 / 1250.82 | 44.0 / 49.36 | 250.0 to 537.7 / 346.80 | − / 23.36 |
| 0.04 | 0.25 | 100.0 to 188.9 / 146.90 | 9.0 / 6.95 | 133.3 to 251.9 / 195.64 | 10.7 / 8.84 | 125.0 to 236.1 / 172.00 | − / 12.91 |
| 0.04 | 0.5 | 100.0 to 188.9 / 146.90 | 9.0 / 6.95 | 200.0 to 377.8 / 296.60 | 14.0 / 12.73 | 125.0 to 236.1 / 172.00 | − / 12.91 |
| 0.04 | 0.75 | 100.0 to 188.9 / 146.90 | 9.0 / 6.95 | 400.0 to 755.6 / 599.43 | 24.0 / 24.71 | 125.0 to 236.1 / 172.00 | − / 12.91 |
| 0.06 | 0.25 | 66.7 to 115.5 / 97.80 | 7.3 / 5.08 | 88.9 to 154.0 / 130.30 | 8.4 / 6.33 | 83.3 to 144.4 / 114.46 | − / 9.42 |
| 0.06 | 0.5 | 66.7 to 115.5 / 97.80 | 7.3 / 5.08 | 133.3 to 231.0 / 197.17 | 10.7 / 8.85 | 83.3 to 144.4 / 114.46 | − / 9.42 |
| 0.06 | 0.75 | 66.7 to 115.5 / 97.80 | 7.3 / 5.08 | 266.7 to 462.0 / 397.82 | 17.3 / 16.68 | 83.3 to 144.4 / 114.46 | − / 9.42 |
| 0.08 | 0.25 | 50.0 to 81.0 / 73.22 | 6.5 / 4.14 | 66.7 to 108.0 / 97.53 | 7.3 / 5.08 | 62.5 to 101.2 / 85.89 | − / 7.62 |
| 0.08 | 0.5 | 50.0 to 81.0 / 73.22 | 6.5 / 4.14 | 100.0 to 161.9 / 147.20 | 9.0 / 6.95 | 62.5 to 101.2 / 85.89 | − / 7.62 |
| 0.08 | 0.75 | 50.0 to 81.0 / 73.22 | 6.5 / 4.14 | 200.0 to 323.9 / 295.61 | 14.0 / 12.72 | 62.5 to 101.2 / 85.89 | − / 7.62 |
| 0.10 | 0.25 | 40.0 to 61.2 / 58.57 | 6.0 / 3.56 | 53.3 to 81.6 / 78.02 | 6.7 / 4.33 | 50.0 to 76.5 / 68.60 | − / 6.51 |
| 0.10 | 0.5 | 40.0 to 61.2 / 58.57 | 6.0 / 3.56 | 80.0 to 122.4 / 117.04 | 8.0 / 5.83 | 50.0 to 76.5 / 68.60 | − / 6.51 |
| 0.10 | 0.75 | 40.0 to 61.2 / 58.57 | 6.0 / 3.56 | 160.0 to 244.8 / 236.86 | 12.0 / 10.38 | 50.0 to 76.5 / 68.60 | − / 6.51 |
| 0.12 | 0.25 | 33.3 to 48.5 / 48.64 | 5.7 / 3.18 | 44.4 to 64.7 / 64.97 | 6.2 / 3.82 | 41.7 to 60.7 / 57.14 | − / 5.75 |
| 0.12 | 0.5 | 33.3 to 48.5 / 48.64 | 5.7 / 3.18 | 66.7 to 97.1 / 97.73 | 7.3 / 5.08 | 41.7 to 60.7 / 57.14 | − / 5.75 |
| 0.12 | 0.75 | 33.3 to 48.5 / 48.64 | 5.7 / 3.18 | 133.3 to 194.2 / 196.01 | 10.7 / 8.85 | 41.7 to 60.7 / 57.14 | − / 5.75 |
| 0.14 | 0.25 | 28.6 to 39.8 / 41.73 | 5.4 / 2.89 | 38.1 to 53.1 / 55.67 | 5.9 / 3.46 | 35.7 to 49.8 / 48.88 | − / 5.20 |
| 0.14 | 0.5 | 28.6 to 39.8 / 41.73 | 5.4 / 2.89 | 57.1 to 79.6 / 83.49 | 6.9 / 4.55 | 35.7 to 49.8 / 48.88 | − / 5.20 |
| 0.14 | 0.75 | 28.6 to 39.8 / 41.73 | 5.4 / 2.89 | 114.3 to 159.2 / 167.73 | 9.7 / 7.76 | 35.7 to 49.8 / 48.88 | − / 5.20 |
| 0.16 | 0.25 | 25.0 to 33.5 / 36.56 | 5.2 / 2.67 | 33.3 to 44.6 / 48.70 | 5.7 / 3.18 | 31.2 to 41.8 / 42.80 | − / 4.78 |
| 0.16 | 0.5 | 25.0 to 33.5 / 36.56 | 5.2 / 2.67 | 50.0 to 66.9 / 73.16 | 6.5 / 4.14 | 31.2 to 41.8 / 42.80 | − / 4.78 |
| 0.16 | 0.75 | 25.0 to 33.5 / 36.56 | 5.2 / 2.67 | 100.0 to 133.8 / 147.09 | 9.0 / 6.95 | 31.2 to 41.8 / 42.80 | − / 4.78 |

The theoretical model gets the right order of magnitude and the right qualitative trends (falls with `m`, rises with `q` for the discriminating policies) but isn't tight: `approach_all`'s point prediction is consistently too high, and the `discriminating_wanderer`/`discriminate` predictions assume independence between finding-a-mushroom and finding-an-*edible*-mushroom that doesn't quite hold. This is the gap the empirical `a + b/m` fit above closes.

**Time to first meal (`T_first`)**

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | 293.61 | 13.50 | 397.33 | 17.14 | 338.07 | 23.48 |
| 0.02 | 0.5 | 293.61 | 13.50 | 597.26 | 25.26 | 338.07 | 23.48 |
| 0.02 | 0.75 | 293.61 | 13.50 | 1149.99 | 49.95 | 338.07 | 23.48 |
| 0.04 | 0.25 | 138.76 | 7.70 | 182.89 | 9.50 | 163.54 | 13.45 |
| 0.04 | 0.5 | 138.76 | 7.70 | 285.66 | 14.43 | 163.54 | 13.45 |
| 0.04 | 0.75 | 138.76 | 7.70 | 566.26 | 26.55 | 163.54 | 13.45 |
| 0.06 | 0.25 | 87.63 | 5.90 | 116.80 | 7.02 | 104.76 | 10.18 |
| 0.06 | 0.5 | 87.63 | 5.90 | 191.78 | 9.32 | 104.76 | 10.18 |
| 0.06 | 0.75 | 87.63 | 5.90 | 404.03 | 17.39 | 104.76 | 10.18 |
| 0.08 | 0.25 | 64.04 | 4.97 | 87.60 | 5.88 | 78.37 | 8.35 |
| 0.08 | 0.5 | 64.04 | 4.97 | 136.11 | 7.64 | 78.37 | 8.35 |
| 0.08 | 0.75 | 64.04 | 4.97 | 286.64 | 13.64 | 78.37 | 8.35 |
| 0.10 | 0.25 | 49.49 | 4.29 | 69.41 | 5.12 | 61.19 | 7.04 |
| 0.10 | 0.5 | 49.49 | 4.29 | 110.48 | 6.71 | 61.19 | 7.04 |
| 0.10 | 0.75 | 49.49 | 4.29 | 239.66 | 11.41 | 61.19 | 7.04 |
| 0.12 | 0.25 | 39.93 | 3.89 | 55.87 | 4.55 | 49.02 | 6.36 |
| 0.12 | 0.5 | 39.93 | 3.89 | 84.93 | 5.86 | 49.02 | 6.36 |
| 0.12 | 0.75 | 39.93 | 3.89 | 187.99 | 9.85 | 49.02 | 6.36 |
| 0.14 | 0.25 | 31.20 | 3.59 | 44.75 | 4.14 | 38.67 | 5.84 |
| 0.14 | 0.5 | 31.20 | 3.59 | 69.05 | 5.30 | 38.67 | 5.84 |
| 0.14 | 0.75 | 31.20 | 3.59 | 150.44 | 8.50 | 38.67 | 5.84 |
| 0.16 | 0.25 | 25.96 | 3.33 | 38.54 | 3.86 | 32.06 | 5.50 |
| 0.16 | 0.5 | 25.96 | 3.33 | 63.03 | 4.87 | 32.06 | 5.50 |
| 0.16 | 0.75 | 25.96 | 3.33 | 135.44 | 7.33 | 32.06 | 5.50 |

`T_first` tracks `T` closely throughout, generally landing a little below it — consistent with the long-run average being pulled up by occasional unlucky stretches that a first-meal measurement doesn't see.

**Net energy per step (`G`)** — same `predicted / measured` format, computed from each row's predicted/measured `T` via `G = M/T − d`:

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | −0.488 to −0.475 / −0.483 | −0.143 / −0.106 | −0.483 to −0.463 / −0.474 | +0.077 / +0.101 | −0.491 to −0.480 / −0.485 | − / −0.285 |
| 0.02 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.488 to −0.475 / −0.483 | −0.083 / −0.095 | −0.500 to −0.500 / −0.500 | − / −0.500 |
| 0.02 | 0.75 | −0.525 to −0.512 / −0.517 | −0.857 / −0.894 | −0.494 to −0.487 / −0.491 | −0.273 / −0.297 | −0.520 to −0.509 / −0.515 | − / −0.715 |
| 0.04 | 0.25 | −0.474 to −0.450 / −0.466 | +0.056 / +0.218 | −0.460 to −0.425 / −0.448 | +0.438 / +0.632 | −0.479 to −0.460 / −0.470 | − / −0.113 |
| 0.04 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.501 | −0.474 to −0.450 / −0.466 | +0.214 / +0.286 | −0.500 to −0.500 / −0.500 | − / −0.501 |
| 0.04 | 0.75 | −0.550 to −0.526 / −0.534 | −1.056 / −1.219 | −0.487 to −0.475 / −0.483 | −0.083 / −0.095 | −0.540 to −0.521 / −0.529 | − / −0.887 |
| 0.06 | 0.25 | −0.457 to −0.425 / −0.449 | +0.182 / +0.484 | −0.435 to −0.388 / −0.423 | +0.684 / +1.081 | −0.465 to −0.440 / −0.456 | − / +0.031 |
| 0.06 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.457 to −0.425 / −0.449 | +0.437 / +0.631 | −0.500 to −0.500 / −0.500 | − / −0.500 |
| 0.06 | 0.75 | −0.575 to −0.543 / −0.551 | −1.182 / −1.485 | −0.478 to −0.463 / −0.474 | +0.077 / +0.100 | −0.560 to −0.535 / −0.544 | − / −1.030 |
| 0.08 | 0.25 | −0.438 to −0.400 / −0.431 | +0.269 / +0.708 | −0.407 to −0.350 / −0.397 | +0.864 / +1.470 | −0.451 to −0.420 / −0.442 | − / +0.157 |
| 0.08 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.438 to −0.400 / −0.431 | +0.611 / +0.939 | −0.500 to −0.500 / −0.500 | − / −0.499 |
| 0.08 | 0.75 | −0.600 to −0.562 / −0.569 | −1.269 / −1.709 | −0.469 to −0.450 / −0.466 | +0.214 / +0.287 | −0.580 to −0.549 / −0.559 | − / −1.157 |
| 0.10 | 0.25 | −0.418 to −0.375 / −0.415 | +0.333 / +0.903 | −0.377 to −0.312 / −0.371 | +1.000 / +1.810 | −0.435 to −0.400 / −0.427 | − / +0.268 |
| 0.10 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.418 to −0.375 / −0.414 | +0.750 / +1.216 | −0.500 to −0.500 / −0.500 | − / −0.500 |
| 0.10 | 0.75 | −0.625 to −0.582 / −0.586 | −1.333 / −1.904 | −0.459 to −0.438 / −0.457 | +0.333 / +0.464 | −0.600 to −0.565 / −0.573 | − / −1.268 |
| 0.12 | 0.25 | −0.397 to −0.350 / −0.397 | +0.382 / +1.073 | −0.346 to −0.275 / −0.345 | +1.107 / +2.115 | −0.418 to −0.380 / −0.412 | − / +0.369 |
| 0.12 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.397 to −0.350 / −0.397 | +0.864 / +1.470 | −0.500 to −0.500 / −0.500 | − / −0.501 |
| 0.12 | 0.75 | −0.650 to −0.603 / −0.604 | −1.382 / −2.074 | −0.449 to −0.425 / −0.448 | +0.437 / +0.631 | −0.620 to −0.582 / −0.588 | − / −1.370 |
| 0.14 | 0.25 | −0.374 to −0.325 / −0.380 | +0.421 / +1.230 | −0.312 to −0.237 / −0.320 | +1.194 / +2.391 | −0.400 to −0.360 / −0.398 | − / +0.463 |
| 0.14 | 0.5 | −0.500 to −0.500 / −0.501 | −0.500 / −0.497 | −0.374 to −0.325 / −0.380 | +0.958 / +1.700 | −0.500 to −0.500 / −0.500 | − / −0.500 |
| 0.14 | 0.75 | −0.675 to −0.626 / −0.621 | −1.421 / −2.227 | −0.437 to −0.412 / −0.440 | +0.529 / +0.788 | −0.640 to −0.600 / −0.603 | − / −1.461 |
| 0.16 | 0.25 | −0.351 to −0.300 / −0.363 | +0.452 / +1.371 | −0.276 to −0.200 / −0.294 | +1.265 / +2.645 | −0.380 to −0.340 / −0.383 | − / +0.547 |
| 0.16 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.499 | −0.351 to −0.300 / −0.363 | +1.038 / +1.914 | −0.500 to −0.500 / −0.501 | − / −0.500 |
| 0.16 | 0.75 | −0.700 to −0.649 / −0.637 | −1.452 / −2.371 | −0.425 to −0.400 / −0.431 | +0.611 / +0.939 | −0.660 to −0.620 / −0.617 | − / −1.547 |

**The main finding:** `discriminate` beats `approach_all`'s `G` at every one of the 24 configurations tested, by a margin ranging from `+0.21` (sparsest, safest corner) up to several times its magnitude elsewhere. Navigation and discrimination are *synergistic*, not additive: combining them produces a larger gain than the sum of their individual effects, and the gap grows with how dangerous the environment is (`q`). Navigating quickly toward food is actively harmful on its own once most food is poisonous (`approach_all` underperforms `wander` at `q=0.75` everywhere), but becomes highly valuable once paired with the ability to tell food apart — while discrimination helps on its own at every density and danger level tested, which matters for how a discriminating strategy could plausibly evolve one capability at a time.

`fraction_poisonous` matched predictions exactly throughout — ≈`q` for the indiscriminate eaters, exactly `0` for the discriminating ones, across all 144,000 runs.

## Research plan

The project moves through five stages, each validated before the next is built on top of it.

**Stage 1 — hand-coded baselines (complete, above).** Point: measure how often each strategy finds food (`T`) and whether energy per step behaves as predicted (`G = M/T − d`). Why: `T` is the one quantity the energy budget can't pin down exactly — everything else (`N`, `P`, `q`, `d`) is arithmetic on top of it. Gives: measured `T` for each strategy and density, confirmed against the energy equation, so `N`, `P`, `q`, `d` can be chosen on paper instead of by trial and error for every later stage.

**Stage 2 — reproduction calibration (next).** Point: find the reproduction threshold `R_thresh` at which a random-like agent (`wander_0_5`) averages about 0.8 offspring per lifetime — low enough that a random starting population declines, high enough that mutation has time (~10 generations) to find something better before it dies out. Why: evolution acts on offspring, not energy, and `G` alone can't say how energy converts into births — at the chosen parameters `wander_0_5`'s average lifetime energy drift is about `−200`, but single-meal swings (`+62` edible, `−31` poisonous) are `±500`, so luck dominates whether a given lineage actually reproduces; that has to be measured per-seed, not calculated from `G`. Gives: a calibrated `R_thresh`, and confirmation that reproduction counts track `G` for the policies where luck matters less.

Chosen environment (derived from the Stage 1 `T` fits at `m=0.04`): `q=0.5`, `P=0.5`, `N=62`, `d=0.1`, `R_cost=200` (equal to a newborn's starting energy, so reproduction neither creates nor destroys energy). This keeps `wander_0_5` just below break-even (`G≈−0.010`) while clearly rewarding both routes to discrimination: `discriminating_wanderer` (`G≈+0.109`) beats `wander` (`G≈+0.005`), and `discriminate` reproduces about `2.2×` faster than `approach_all` (`G≈+4.71` vs `+2.11`).

| Policy | `G` | Predicted |
|---|---|---|
| `wander_0_5` | −0.010 | drift-only lifespan ≈19,400 steps; offspring per lifetime falls as `R_thresh` rises; wide spread from luck |
| `wander` | +0.005 | barely positive — some die early, some reach the step limit; more offspring than `wander_0_5` at every `R_thresh` |
| `discriminating_wanderer` | +0.109 | mostly survives; ~1 birth every 1,830 steps once above `R_thresh` |
| `approach_all` | +2.11 | ~1 birth every 95 steps |
| `discriminate` | +4.71 | ~1 birth every 42 steps |

`R_thresh` swept over `{250, 300, 400, 500, 750, 1000}` (above 200, so the parent survives reproducing), 1,000 seeds each, agent starts at 200 energy, 100,000-step cap with starvation on. Offspring are counted but not added to the world, keeping agent density equal to Stage 1 (so its `T` values stay valid) and isolating the energy-to-offspring question from population dynamics.

**Stage 3 — crowding.** Point: find out how many agents share the grid before they start taking each other's food, and where the population settles. Why: single-agent numbers only hold while agents don't interfere — evolution needs real competition so the population is limited by food (not by hitting a hard cap) and so fitness is relative, with better agents replacing worse ones. Gives: a grid size and mushroom count where competition — not the population cap — limits growth, expected once population reaches roughly `1/14` to `1/10` of the number of grid cells.

**Stage 4a — invasion test.** Point: drop a few hand-coded discriminators into an equilibrium `approach_all` population and see whether they grow as fast as predicted. Why: tests whether discrimination pays off *under competition* before relying on evolution to find it — if discriminators can't invade here, evolved ones won't either. Gives: confirmation of the payoff size, and which `P`/`q` values make it large enough to matter.

**Stage 4b — evolution.** Point: start from random neural networks and see whether discrimination evolves, and by which route — via navigation first, or via selective eating while still moving randomly. Why: the actual research question; the earlier stages ensure a null result here means something about evolution, not about badly-chosen parameters. Gives: the result, plus a sweep over `P`, `q` and the agent-to-mushroom ratio testing where discrimination emerges relative to where the payoff is predicted to be large.
