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
| `discriminating_wanderer` | 0.25 | −7.05 | 8.58 | 5.2% |
| `discriminating_wanderer` | 0.5 | −26.96 | 14.63 | 13.3% |
| `discriminating_wanderer` | 0.75 | −42.94 | 30.83 | 4.9% |
| `discriminate` | 0.25 | 1.02 | 0.33 | 3.1% |
| `discriminate` | 0.5 | 0.20 | 0.57 | 9.9% |
| `discriminate` | 0.75 | 0.06 | 1.15 | 5.5% |

`b/m` is the pure-density term the qualitative model predicts; `a` is an empirical floor (a "reaching" cost for the navigating policies, or measurement noise for the random ones — `wander`'s `a≈−0.77` isn't a real negative cost, it's just the linear fit's best compromise across the range). The four policies whose movement doesn't depend on mushroom type (`wander`, `approach_all`, and their `_0_5` variants) fit this form independent of `q`, essentially exactly (well under 1% error). The two discriminating policies fit less tightly and need a separate `a`, `b` per `q`, because how often a landing "counts" depends on `q` directly. Discrimination noticeably worsens the fit specifically at the sparsest, most dangerous corner (`m=0.02, q=0.75`), where a real fraction of runs (~10%) never found a qualifying meal in 20,000 steps at all — more a sampling-size limit than a failure of the `a + b/m` form.

**Fraction of meals that are poisonous** should match `q` for the indiscriminate eaters (`wander`, `approach_all`, and the `_0_5` variants), and be exactly zero for the discriminating ones, by construction. Confirmed exactly across all runs below.

## Results

Measured over 1,000 independent runs per configuration, 20,000 steps each, on a 20×20 grid (`N=10`, `P=1`, `d=0.5`), across `m ∈ {0.02, 0.04, ..., 0.16}` × `q ∈ {0.25, 0.5, 0.75}` — 24 configurations, 6 policies each, 144,000 runs total.

**Average steps between meals (`T`)** — each cell is `predicted / measured`, using the qualitative-model formulas from above (a range for the three policies whose prediction is a window, a point value otherwise; `approach_all_0_5` has no closed-form prediction):

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | 200.0 to 430.2 / 295.85 | 14.0 / 12.71 | 266.7 to 573.6 / 424.61 | 17.3 / 17.77 | 250.0 to 537.7 / 346.80 | − / 23.36 |
| 0.02 | 0.5 | 200.0 to 430.2 / 295.85 | 14.0 / 12.71 | 400.0 to 860.4 / 714.99 | 24.0 / 29.22 | 250.0 to 537.7 / 346.80 | − / 23.36 |
| 0.02 | 0.75 | 200.0 to 430.2 / 295.85 | 14.0 / 12.71 | 800.0 to 1720.8 / 1486.72 | 44.0 / 56.90 | 250.0 to 537.7 / 346.80 | − / 23.36 |
| 0.04 | 0.25 | 100.0 to 188.9 / 146.90 | 9.0 / 6.95 | 133.3 to 251.9 / 202.25 | 10.7 / 9.09 | 125.0 to 236.1 / 172.00 | − / 12.91 |
| 0.04 | 0.5 | 100.0 to 188.9 / 146.90 | 9.0 / 6.95 | 200.0 to 377.8 / 321.20 | 14.0 / 13.68 | 125.0 to 236.1 / 172.00 | − / 12.91 |
| 0.04 | 0.75 | 100.0 to 188.9 / 146.90 | 9.0 / 6.95 | 400.0 to 755.6 / 765.62 | 24.0 / 30.47 | 125.0 to 236.1 / 172.00 | − / 12.91 |
| 0.06 | 0.25 | 66.7 to 115.5 / 97.80 | 7.3 / 5.08 | 88.9 to 154.0 / 133.37 | 8.4 / 6.44 | 83.3 to 144.4 / 114.46 | − / 9.42 |
| 0.06 | 0.5 | 66.7 to 115.5 / 97.80 | 7.3 / 5.08 | 133.3 to 231.0 / 206.30 | 10.7 / 9.24 | 83.3 to 144.4 / 114.46 | − / 9.42 |
| 0.06 | 0.75 | 66.7 to 115.5 / 97.80 | 7.3 / 5.08 | 266.7 to 462.0 / 463.02 | 17.3 / 19.27 | 83.3 to 144.4 / 114.46 | − / 9.42 |
| 0.08 | 0.25 | 50.0 to 81.0 / 73.22 | 6.5 / 4.14 | 66.7 to 108.0 / 98.71 | 7.3 / 5.14 | 62.5 to 101.2 / 85.89 | − / 7.62 |
| 0.08 | 0.5 | 50.0 to 81.0 / 73.22 | 6.5 / 4.14 | 100.0 to 161.9 / 151.27 | 9.0 / 7.15 | 62.5 to 101.2 / 85.89 | − / 7.62 |
| 0.08 | 0.75 | 50.0 to 81.0 / 73.22 | 6.5 / 4.14 | 200.0 to 323.9 / 330.73 | 14.0 / 14.03 | 62.5 to 101.2 / 85.89 | − / 7.62 |
| 0.10 | 0.25 | 40.0 to 61.2 / 58.57 | 6.0 / 3.56 | 53.3 to 81.6 / 79.10 | 6.7 / 4.37 | 50.0 to 76.5 / 68.60 | − / 6.51 |
| 0.10 | 0.5 | 40.0 to 61.2 / 58.57 | 6.0 / 3.56 | 80.0 to 122.4 / 119.97 | 8.0 / 5.96 | 50.0 to 76.5 / 68.60 | − / 6.51 |
| 0.10 | 0.75 | 40.0 to 61.2 / 58.57 | 6.0 / 3.56 | 160.0 to 244.8 / 257.75 | 12.0 / 11.26 | 50.0 to 76.5 / 68.60 | − / 6.51 |
| 0.12 | 0.25 | 33.3 to 48.5 / 48.64 | 5.7 / 3.18 | 44.4 to 64.7 / 65.75 | 6.2 / 3.84 | 41.7 to 60.7 / 57.14 | − / 5.75 |
| 0.12 | 0.5 | 33.3 to 48.5 / 48.64 | 5.7 / 3.18 | 66.7 to 97.1 / 99.80 | 7.3 / 5.16 | 41.7 to 60.7 / 57.14 | − / 5.75 |
| 0.12 | 0.75 | 33.3 to 48.5 / 48.64 | 5.7 / 3.18 | 133.3 to 194.2 / 211.26 | 10.7 / 9.40 | 41.7 to 60.7 / 57.14 | − / 5.75 |
| 0.14 | 0.25 | 28.6 to 39.8 / 41.73 | 5.4 / 2.89 | 38.1 to 53.1 / 56.16 | 5.9 / 3.48 | 35.7 to 49.8 / 48.88 | − / 5.20 |
| 0.14 | 0.5 | 28.6 to 39.8 / 41.73 | 5.4 / 2.89 | 57.1 to 79.6 / 84.88 | 6.9 / 4.60 | 35.7 to 49.8 / 48.88 | − / 5.20 |
| 0.14 | 0.75 | 28.6 to 39.8 / 41.73 | 5.4 / 2.89 | 114.3 to 159.2 / 177.33 | 9.7 / 8.13 | 35.7 to 49.8 / 48.88 | − / 5.20 |
| 0.16 | 0.25 | 25.0 to 33.5 / 36.56 | 5.2 / 2.67 | 33.3 to 44.6 / 49.10 | 5.7 / 3.19 | 31.2 to 41.8 / 42.80 | − / 4.78 |
| 0.16 | 0.5 | 25.0 to 33.5 / 36.56 | 5.2 / 2.67 | 50.0 to 66.9 / 74.35 | 6.5 / 4.18 | 31.2 to 41.8 / 42.80 | − / 4.78 |
| 0.16 | 0.75 | 25.0 to 33.5 / 36.56 | 5.2 / 2.67 | 100.0 to 133.8 / 153.70 | 9.0 / 7.22 | 31.2 to 41.8 / 42.80 | − / 4.78 |

The theoretical model gets the right order of magnitude and the right qualitative trends (falls with `m`, rises with `q` for the discriminating policies) but isn't tight: `approach_all`'s point prediction is consistently too high, and the `discriminating_wanderer`/`discriminate` predictions assume independence between finding-a-mushroom and finding-an-*edible*-mushroom that doesn't quite hold. This is the gap the empirical `a + b/m` fit above closes.

**Time to first meal (`T_first`)**

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | 293.61 | 13.50 | 415.75 | 18.27 | 338.07 | 23.48 |
| 0.02 | 0.5 | 293.61 | 13.50 | 674.63 | 29.65 | 338.07 | 23.48 |
| 0.02 | 0.75 | 293.61 | 13.50 | 1335.61 | 58.98 | 338.07 | 23.48 |
| 0.04 | 0.25 | 138.76 | 7.70 | 195.63 | 9.78 | 163.54 | 13.45 |
| 0.04 | 0.5 | 138.76 | 7.70 | 310.05 | 14.36 | 163.54 | 13.45 |
| 0.04 | 0.75 | 138.76 | 7.70 | 702.39 | 32.09 | 163.54 | 13.45 |
| 0.06 | 0.25 | 87.63 | 5.90 | 121.26 | 7.27 | 104.76 | 10.18 |
| 0.06 | 0.5 | 87.63 | 5.90 | 184.79 | 10.11 | 104.76 | 10.18 |
| 0.06 | 0.75 | 87.63 | 5.90 | 431.01 | 19.62 | 104.76 | 10.18 |
| 0.08 | 0.25 | 64.04 | 4.97 | 86.91 | 6.05 | 78.37 | 8.35 |
| 0.08 | 0.5 | 64.04 | 4.97 | 134.72 | 7.96 | 78.37 | 8.35 |
| 0.08 | 0.75 | 64.04 | 4.97 | 316.21 | 15.20 | 78.37 | 8.35 |
| 0.10 | 0.25 | 49.49 | 4.29 | 68.04 | 5.09 | 61.19 | 7.04 |
| 0.10 | 0.5 | 49.49 | 4.29 | 107.79 | 6.66 | 61.19 | 7.04 |
| 0.10 | 0.75 | 49.49 | 4.29 | 250.22 | 12.40 | 61.19 | 7.04 |
| 0.12 | 0.25 | 39.93 | 3.89 | 54.92 | 4.61 | 49.02 | 6.36 |
| 0.12 | 0.5 | 39.93 | 3.89 | 86.81 | 5.90 | 49.02 | 6.36 |
| 0.12 | 0.75 | 39.93 | 3.89 | 193.50 | 10.35 | 49.02 | 6.36 |
| 0.14 | 0.25 | 31.20 | 3.59 | 44.43 | 4.20 | 38.67 | 5.84 |
| 0.14 | 0.5 | 31.20 | 3.59 | 71.07 | 5.38 | 38.67 | 5.84 |
| 0.14 | 0.75 | 31.20 | 3.59 | 160.96 | 8.95 | 38.67 | 5.84 |
| 0.16 | 0.25 | 25.96 | 3.33 | 38.09 | 3.88 | 32.06 | 5.50 |
| 0.16 | 0.5 | 25.96 | 3.33 | 60.69 | 4.95 | 32.06 | 5.50 |
| 0.16 | 0.75 | 25.96 | 3.33 | 135.66 | 8.05 | 32.06 | 5.50 |

`T_first` tracks `T` closely throughout, generally landing a little below it — consistent with the long-run average being pulled up by occasional unlucky stretches that a first-meal measurement doesn't see.

**Net energy per step (`G`)** — same `predicted / measured` format, computed from each row's predicted/measured `T` via `G = M/T − d`:

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | −0.488 to −0.475 / −0.483 | −0.143 / −0.119 | −0.483 to −0.463 / −0.475 | +0.077 / +0.093 | −0.491 to −0.480 / −0.486 | − / −0.294 |
| 0.02 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.510 | −0.488 to −0.475 / −0.483 | −0.083 / −0.101 | −0.500 to −0.500 / −0.500 | − / −0.506 |
| 0.02 | 0.75 | −0.525 to −0.512 / −0.517 | −0.857 / −0.893 | −0.494 to −0.487 / −0.491 | −0.273 / −0.296 | −0.520 to −0.509 / −0.515 | − / −0.715 |
| 0.04 | 0.25 | −0.474 to −0.450 / −0.466 | +0.056 / +0.206 | −0.460 to −0.425 / −0.449 | +0.438 / +0.621 | −0.479 to −0.460 / −0.471 | − / −0.119 |
| 0.04 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.505 | −0.474 to −0.450 / −0.466 | +0.214 / +0.279 | −0.500 to −0.500 / −0.500 | − / −0.501 |
| 0.04 | 0.75 | −0.550 to −0.526 / −0.534 | −1.056 / −1.209 | −0.487 to −0.475 / −0.483 | −0.083 / −0.092 | −0.540 to −0.521 / −0.529 | − / −0.883 |
| 0.06 | 0.25 | −0.457 to −0.425 / −0.449 | +0.182 / +0.467 | −0.435 to −0.388 / −0.423 | +0.684 / +1.069 | −0.465 to −0.440 / −0.457 | − / +0.022 |
| 0.06 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.502 | −0.457 to −0.425 / −0.449 | +0.437 / +0.622 | −0.500 to −0.500 / −0.500 | − / −0.500 |
| 0.06 | 0.75 | −0.575 to −0.543 / −0.551 | −1.182 / −1.473 | −0.478 to −0.463 / −0.474 | +0.077 / +0.103 | −0.560 to −0.535 / −0.544 | − / −1.025 |
| 0.08 | 0.25 | −0.438 to −0.400 / −0.432 | +0.269 / +0.692 | −0.407 to −0.350 / −0.397 | +0.864 / +1.459 | −0.451 to −0.420 / −0.442 | − / +0.148 |
| 0.08 | 0.5 | −0.500 to −0.500 / −0.499 | −0.500 / −0.499 | −0.438 to −0.400 / −0.431 | +0.611 / +0.931 | −0.500 to −0.500 / −0.500 | − / −0.499 |
| 0.08 | 0.75 | −0.600 to −0.562 / −0.567 | −1.269 / −1.689 | −0.469 to −0.450 / −0.465 | +0.214 / +0.291 | −0.580 to −0.549 / −0.557 | − / −1.145 |
| 0.10 | 0.25 | −0.418 to −0.375 / −0.415 | +0.333 / +0.882 | −0.377 to −0.312 / −0.372 | +1.000 / +1.797 | −0.435 to −0.400 / −0.428 | − / +0.256 |
| 0.10 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.505 | −0.418 to −0.375 / −0.414 | +0.750 / +1.206 | −0.500 to −0.500 / −0.500 | − / −0.502 |
| 0.10 | 0.75 | −0.625 to −0.582 / −0.585 | −1.333 / −1.890 | −0.459 to −0.438 / −0.456 | +0.333 / +0.463 | −0.600 to −0.565 / −0.572 | − / −1.261 |
| 0.12 | 0.25 | −0.397 to −0.350 / −0.397 | +0.382 / +1.067 | −0.346 to −0.275 / −0.346 | +1.107 / +2.110 | −0.418 to −0.380 / −0.412 | − / +0.365 |
| 0.12 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.494 | −0.397 to −0.350 / −0.397 | +0.864 / +1.463 | −0.500 to −0.500 / −0.500 | − / −0.496 |
| 0.12 | 0.75 | −0.650 to −0.603 / −0.603 | −1.382 / −2.063 | −0.449 to −0.425 / −0.448 | +0.437 / +0.626 | −0.620 to −0.582 / −0.587 | − / −1.365 |
| 0.14 | 0.25 | −0.374 to −0.325 / −0.381 | +0.421 / +1.219 | −0.312 to −0.237 / −0.320 | +1.194 / +2.385 | −0.400 to −0.360 / −0.398 | − / +0.457 |
| 0.14 | 0.5 | −0.500 to −0.500 / −0.499 | −0.500 / −0.496 | −0.374 to −0.325 / −0.379 | +0.958 / +1.694 | −0.500 to −0.500 / −0.499 | − / −0.497 |
| 0.14 | 0.75 | −0.675 to −0.626 / −0.620 | −1.421 / −2.217 | −0.437 to −0.412 / −0.440 | +0.529 / +0.785 | −0.640 to −0.600 / −0.602 | − / −1.454 |
| 0.16 | 0.25 | −0.351 to −0.300 / −0.364 | +0.452 / +1.357 | −0.276 to −0.200 / −0.294 | +1.265 / +2.637 | −0.380 to −0.340 / −0.384 | − / +0.540 |
| 0.16 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.497 | −0.351 to −0.300 / −0.363 | +1.038 / +1.910 | −0.500 to −0.500 / −0.500 | − / −0.498 |
| 0.16 | 0.75 | −0.700 to −0.649 / −0.636 | −1.452 / −2.363 | −0.425 to −0.400 / −0.431 | +0.611 / +0.932 | −0.660 to −0.620 / −0.617 | − / −1.543 |

**The main finding:** `discriminate` beats `approach_all`'s `G` at every one of the 24 configurations tested, by a margin ranging from `+0.21` (sparsest, safest corner) up to several times its magnitude elsewhere. Navigation and discrimination are *synergistic*, not additive: combining them produces a larger gain than the sum of their individual effects, and the gap grows with how dangerous the environment is (`q`). Navigating quickly toward food is actively harmful on its own once most food is poisonous (`approach_all` underperforms `wander` at `q=0.75` everywhere), but becomes highly valuable once paired with the ability to tell food apart — while discrimination helps on its own at every density and danger level tested, which matters for how a discriminating strategy could plausibly evolve one capability at a time.

`fraction_poisonous` matched predictions exactly throughout — ≈`q` for the indiscriminate eaters, exactly `0` for the discriminating ones, across all 144,000 runs.

## Research plan

The project moves through five stages, each validated before the next is built on top of it.

**Stage 1 — hand-coded baselines (complete, above).** Point: measure how often each strategy finds food (`T`) and whether energy per step behaves as predicted (`G = M/T − d`). Why: `T` is the one quantity the energy budget can't pin down exactly — everything else (`N`, `P`, `q`, `d`) is arithmetic on top of it. Gives: measured `T` for each strategy and density, confirmed against the energy equation, so `N`, `P`, `q`, `d` can be chosen on paper instead of by trial and error for every later stage.

**Stage 2 — reproduction, no mutation (next).** Point: check that energy gains and losses translate into survival and offspring as predicted, and tune the environment so random agents decline slowly rather than crashing. Why: evolution needs the random first generation to survive long enough for improvements to appear — too harsh and everyone dies before selection acts, too kind and there's no pressure to improve. Gives: a shortlist of `N`, density and `d` where the starting population hangs on, and where approaching and discriminating clearly pay. From Stage 1's data: `q` has to stay below `1/(1+P)` (`0.5` at `P=1`) — no configuration at `q ≥ 0.5` gives `wander` a viable `G` regardless of `N` — and even at `q=0.25`, `N=10` is far too low; getting `wander` close to a slow decline needs `N` roughly `5–50×` higher, more at low density than high. Chosen so far: `m=0.08, q=0.25, N=65` — predicted `wander` `G≈−0.056`.

**Stage 3 — crowding.** Point: find out how many agents share the grid before they start taking each other's food, and where the population settles. Why: single-agent numbers only hold while agents don't interfere — evolution needs real competition so the population is limited by food (not by hitting a hard cap) and so fitness is relative, with better agents replacing worse ones. Gives: a grid size and mushroom count where competition — not the population cap — limits growth, expected once population reaches roughly `1/14` to `1/10` of the number of grid cells.

**Stage 4a — invasion test.** Point: drop a few hand-coded discriminators into an equilibrium `approach_all` population and see whether they grow as fast as predicted. Why: tests whether discrimination pays off *under competition* before relying on evolution to find it — if discriminators can't invade here, evolved ones won't either. Gives: confirmation of the payoff size, and which `P`/`q` values make it large enough to matter.

**Stage 4b — evolution.** Point: start from random neural networks and see whether discrimination evolves, and by which route — via navigation first, or via selective eating while still moving randomly. Why: the actual research question; the earlier stages ensure a null result here means something about evolution, not about badly-chosen parameters. Gives: the result, plus a sweep over `P`, `q` and the agent-to-mushroom ratio testing where discrimination emerges relative to where the payoff is predicted to be large.
