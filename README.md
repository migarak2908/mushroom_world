# Mushroom Foraging Testbed

A JAX testbed for **open-ended, non-episodic evolution** of foraging agents on a grid.

## What this experiment is for

**The question.** Under what environmental conditions does an agent evolve to tell food apart from poison (eating edible mushrooms and avoiding poisonous ones by their features), starting from randomly initialised agents, in a population where competition keeps the population turning over?

**Why it's interesting.** Discrimination is a simple case of a general problem: how a capability that pays off only once several pieces are in place (seeing a feature, linking it to an action, acting on it reliably) can be assembled by selection one step at a time. Whether it evolves depends on the environment: how common and how harmful poison is, how scarce food is, how crowded the world is. Too harsh and the random first generation dies before selection can act; too kind and there is no pressure to improve. We want to map where between those extremes discrimination appears, and by which route: learning to approach food first and then becoming selective, or becoming a selective eater while still moving randomly.

**Why open-ended and non-episodic.** There are no generations, resets or external fitness function. Agents live, eat, reproduce and die continuously in a shared world, and fitness is just who leaves more offspring. That is closer to how selection actually works and lets strategies keep competing and replacing each other, but it also means many things can go wrong for mechanical rather than evolutionary reasons (the population crashing, freezing at a cap, energy leaking).

**The approach: predict first, then evolve.** Rather than tuning parameters by trial and error, each stage derives a simple prediction (mainly an energy budget: energy in from meals minus energy spent per step) and tests it with hand-coded agents whose behaviour is fixed and known. Only once the world behaves as predicted are the parameters used for evolution. Then a failure to evolve discrimination says something about evolution, not about a badly chosen environment.

**A reusable testbed.** The code is modular and class-based (world, food sources, observation channels, policies), so the same machinery can be reused for other open-ended evolution experiments.

## Status

| Stage | Status | Key result | Chosen |
|---|---|---|---|
| 1. Hand-coded baselines | complete | `G = M/T − d` holds to ~1% using measured `T`; `T` fits `a + b/m` within 5% for every policy | n/a |
| 2. Reproduction calibration | complete | `wander_0_5` averages 0.84 offspring per lifetime at `R_thresh = 400` | `m=0.04, q=0.5, P=0.5, N=62, d=0.1, R_thresh=400, R_cost=200` |
| 3. Evolution | next | n/a | grid 150 × 150, `A_max = 1,000`, random replacement at the cap |

## The model

- **World:** a toroidal grid. Mushrooms are either edible or poisonous; eating one respawns it instantly at a random empty cell, keeping the population density constant. The number of poisonous mushrooms is fixed per run (`q × nb`), assigned to random positions at reset.
- **Agents:** each step, an agent loses a fixed amount of energy (`d`), can move (turn or step forward) and can choose to eat whatever is on its current cell. Eating an edible mushroom gains energy (`N`); eating a poisonous one loses `P × N`.
- **Perception:** agents see a 3-deep, 5-wide window in front of themselves (their own row plus two ahead), which rotates with the direction they're facing.
- **Known approximation:** if two agents on the same cell both eat on the same step, both are paid for the one mushroom. This adds roughly a few percent of meals at full population and affects all strategies in proportion to how often they eat.

**Symbols**

| Symbol | Meaning |
|---|---|
| `m` | mushroom density: fraction of grid cells holding a mushroom |
| `q` | fraction of mushrooms that are poisonous |
| `N` | energy from an edible mushroom |
| `P` | energy lost from a poisonous one, as a multiple of `N` |
| `d` | energy lost per step |
| `T` | average steps between meals (long-run) |
| `T_first` | steps until the first meal on a fresh grid |
| `M` | average energy per meal |
| `G` | average net energy change per step |
| `E_s` | starting energy (200) |
| `R_thresh` | energy at which an agent reproduces |
| `R_cost` | energy a parent loses per offspring |

## Behaviours compared

| Policy | Movement | Eating |
|---|---|---|
| `wander` | Random | Eats anything |
| `approach_all` | Navigates toward the nearest visible mushroom | Eats anything |
| `discriminating_wanderer` | Random | Eats only if standing on an edible mushroom |
| `discriminate` | Navigates toward the nearest visible *edible* mushroom | Eats only if standing on an edible mushroom |
| `wander_0_5` | Random | Eats with probability 0.5, regardless of what's underfoot |
| `approach_all_0_5` | Navigates toward the nearest visible mushroom | Eats with probability 0.5, regardless of what's underfoot |

The first four span the combinations of "moves with purpose" × "eats with discrimination," which makes it possible to attribute how much of any survival advantage comes from navigation, from diet, or from both together. The last two add an *imperfect* eater on top of `wander`/`approach_all`: a stand-in for how a freshly-initialised, not-yet-selected network behaves, since it hasn't learned to condition eating on what's actually in front of it.

The navigating policies react to the cells straight ahead (centre column) and the two cells on each side in the agent's own row; when nothing is in view they move forward 90% of the time and turn randomly otherwise.

## Code and how to run

| File | Contents |
|---|---|
| `world.py` | grid, movement, observation window, death and reproduction toggles |
| `mushrooms.py`, `food_source.py` | mushroom state, eating, uniform respawn |
| `observation.py` | observation channels |
| `policy.py`, `baselines.py` | policy interface and the hand-coded policies |
| `runner.py`, `metrics.py` | batched runs and summary metrics |
| `scripts/run_baselines.py` | Stage 1 sweep |
| `scripts/run_threshold_sweep.py` | Stage 2 `R_thresh` sweep |
| `archive/`, `mushroom_world.py`, `scripts/sweeps/`, `scripts/analysis/` | earlier model and its experiments |

```
python scripts/run_baselines.py
python scripts/run_threshold_sweep.py
```

## Energy budget

**Meal value.** An indiscriminate eater gets something edible with probability `(1−q)` and something poisonous with probability `q`, so on average:

```
M = N × [(1 − q) − q×P]
```

Positive only when `q < 1/(1+P)`. Past that point, poison is common or severe enough that eating anything you find is a net loss regardless of `N`. A discriminating eater instead gets `M = N` every meal, since it only ever eats edible mushrooms.

**Net energy per step**, for either kind of eater:

```
G = M/T − d
```

Positive `G` means a population would grow; negative means it would shrink. `T` is the one quantity in this budget that depends on the real dynamics of movement and search; Stage 1 measures it.

## Stage 1: hand-coded baselines

**Point:** measure how often each strategy finds food (`T`) and whether energy per step behaves as predicted (`G = M/T − d`). **Why:** `T` is the one quantity the energy budget can't pin down exactly; everything else (`N`, `P`, `q`, `d`) is arithmetic on top of it. **Gives:** measured `T` for each strategy and density, confirmed against the energy equation, so `N`, `P`, `q`, `d` can be chosen on paper for every later stage.

**Setup:** one agent, death and reproduction off, 20×20 grid, `N=10`, `P=1`, `d=0.5`, `m ∈ {0.02, 0.04, ..., 0.16}` × `q ∈ {0.25, 0.5, 0.75}`, 1,000 runs × 20,000 steps per configuration, 6 policies (144,000 runs).

### Predictions for `T`

| Policy | Prediction | Label |
|---|---|---|
| `wander` | between `4/m` (no cell ever revisited) and a random-walk estimate | derived bounds |
| `approach_all` | `1/(4.5m) + 2.4`: about 4.5 new cells checked per step (5 per forward move × 0.9 forward), plus 2.4 steps to walk to a spotted mushroom (ahead: 2; one to the side: 2; two to the side: 3; averaged over the 5 cells, `(2 + 2×2 + 2×3) ÷ 5 = 2.4`) | spotting term derived, reach approximate |
| `discriminate` | `approach_all`'s `T` at edible density `m × (1−q)` | derived |
| `discriminating_wanderer` | `wander`'s `T ÷ (1−q)` | approximate |
| `wander_0_5` | at most `1.25 ×` wander (a mushroom it lands on is eaten before it leaves with chance 0.8) | upper bound |
| `approach_all_0_5` | about `1.9 ×` approach_all (it walks straight off a skipped mushroom, so eats with chance ≈ 0.5 ÷ 0.95) | approximate |

### Results

**Fitted `T ≈ a + b/m`:**

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

**Checks:**

| Check | Result |
|---|---|
| `G = M/T − d` with measured `T` | ✓ e.g. `approach_all`, `m=0.16, q=0.25`: `5/2.67 − 0.5 = +1.37`, measured `+1.37`; `wander`, `m=0.08, q=0.25`: `5/73.22 − 0.5 = −0.432`, measured `−0.431` |
| `G = −d` exactly for indiscriminate eaters at `q=0.5, P=1` (`M = 0`) | ✓ `−0.497` to `−0.501` throughout |
| Fraction of meals poisonous: `≈q` indiscriminate, `0` discriminating | ✓ across all 144,000 runs |
| `approach_all` spotting term `1/(4.5m)` | ✓ fitted `b = 0.23` vs derived `0.222`; reach `a ≈ 1.3` steps |
| `wander` within `4/m` bound and `∝ 1/m` | ✓ `T × m ≈ 5.9` at every density (revisits cost ~1.5× over `4/m`) |
| `discriminate` ≈ `approach_all` at `m(1−q)` | ✓ within 2% (e.g. `m=0.04, q=0.5`: 12.73 vs 12.76) |
| `discriminating_wanderer` ≈ `wander ÷ (1−q)` | ✓ within 1 to 2% except the sparsest corner (`m=0.02, q=0.75`: 1,251 vs 1,183, +6%) |
| `wander_0_5` ≤ 1.25 × `wander` | ✓ `1.17×` at every density, lower because a random walker comes back to mushrooms it skipped |
| `approach_all_0_5` ≈ 1.9 × `approach_all` | ✓ `1.79 to 1.86×` |
| `T_first` vs `T` | `approach_all`: `T_first ≈ T + 1` (the `+1` is the eating step), so lone-agent depletion doesn't affect it. `wander`: long-run `T` is 1 to 41% above `T_first`, rising with density: the lone-agent depletion the plan predicted |

**What it means:** navigating quickly toward food is harmful on its own once most food is poisonous (`approach_all` underperforms `wander` at `q=0.75` everywhere), and valuable once paired with discrimination. `discriminate` beats `approach_all` and `discriminating_wanderer` beats `wander` at every configuration. This is expected by construction, since each uses the same search as its partner and skips poison, gaining `q × P × N ÷ T` per step.

<details>
<summary>Full Stage 1 tables (24 configurations × 6 policies)</summary>

**Average steps between meals (`T`).** Each cell is `predicted / measured`, using the formulas in the predictions table above: a range for `wander`, `discriminating_wanderer` and `wander_0_5`; `1/(4.5m) + 2.4` for `approach_all` (2.4 is the derived reach cost); the same at `m(1−q)` for `discriminate`; and `1.9 ×` that for `approach_all_0_5`.

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | 200.0 to 430.2 / 295.85 | 13.5 / 12.71 | 266.7 to 573.6 / 396.36 | 17.2 / 16.66 | 250.0 to 537.7 / 346.80 | 25.7 / 23.36 |
| 0.02 | 0.5 | 200.0 to 430.2 / 295.85 | 13.5 / 12.71 | 400.0 to 860.4 / 595.90 | 24.6 / 24.72 | 250.0 to 537.7 / 346.80 | 25.7 / 23.36 |
| 0.02 | 0.75 | 200.0 to 430.2 / 295.85 | 13.5 / 12.71 | 800.0 to 1720.8 / 1250.82 | 46.8 / 49.36 | 250.0 to 537.7 / 346.80 | 25.7 / 23.36 |
| 0.04 | 0.25 | 100.0 to 188.9 / 146.90 | 8.0 / 6.95 | 133.3 to 251.9 / 195.64 | 9.8 / 8.84 | 125.0 to 236.1 / 172.00 | 15.1 / 12.91 |
| 0.04 | 0.5 | 100.0 to 188.9 / 146.90 | 8.0 / 6.95 | 200.0 to 377.8 / 296.60 | 13.5 / 12.73 | 125.0 to 236.1 / 172.00 | 15.1 / 12.91 |
| 0.04 | 0.75 | 100.0 to 188.9 / 146.90 | 8.0 / 6.95 | 400.0 to 755.6 / 599.43 | 24.6 / 24.71 | 125.0 to 236.1 / 172.00 | 15.1 / 12.91 |
| 0.06 | 0.25 | 66.7 to 115.5 / 97.80 | 6.1 / 5.08 | 88.9 to 154.0 / 130.30 | 7.3 / 6.33 | 83.3 to 144.4 / 114.46 | 11.6 / 9.42 |
| 0.06 | 0.5 | 66.7 to 115.5 / 97.80 | 6.1 / 5.08 | 133.3 to 231.0 / 197.17 | 9.8 / 8.85 | 83.3 to 144.4 / 114.46 | 11.6 / 9.42 |
| 0.06 | 0.75 | 66.7 to 115.5 / 97.80 | 6.1 / 5.08 | 266.7 to 462.0 / 397.82 | 17.2 / 16.68 | 83.3 to 144.4 / 114.46 | 11.6 / 9.42 |
| 0.08 | 0.25 | 50.0 to 81.0 / 73.22 | 5.2 / 4.14 | 66.7 to 108.0 / 97.53 | 6.1 / 5.08 | 62.5 to 101.2 / 85.89 | 9.8 / 7.62 |
| 0.08 | 0.5 | 50.0 to 81.0 / 73.22 | 5.2 / 4.14 | 100.0 to 161.9 / 147.20 | 8.0 / 6.95 | 62.5 to 101.2 / 85.89 | 9.8 / 7.62 |
| 0.08 | 0.75 | 50.0 to 81.0 / 73.22 | 5.2 / 4.14 | 200.0 to 323.9 / 295.61 | 13.5 / 12.72 | 62.5 to 101.2 / 85.89 | 9.8 / 7.62 |
| 0.10 | 0.25 | 40.0 to 61.2 / 58.57 | 4.6 / 3.56 | 53.3 to 81.6 / 78.02 | 5.4 / 4.33 | 50.0 to 76.5 / 68.60 | 8.8 / 6.51 |
| 0.10 | 0.5 | 40.0 to 61.2 / 58.57 | 4.6 / 3.56 | 80.0 to 122.4 / 117.04 | 6.8 / 5.83 | 50.0 to 76.5 / 68.60 | 8.8 / 6.51 |
| 0.10 | 0.75 | 40.0 to 61.2 / 58.57 | 4.6 / 3.56 | 160.0 to 244.8 / 236.86 | 11.3 / 10.38 | 50.0 to 76.5 / 68.60 | 8.8 / 6.51 |
| 0.12 | 0.25 | 33.3 to 48.5 / 48.64 | 4.3 / 3.18 | 44.4 to 64.7 / 64.97 | 4.9 / 3.82 | 41.7 to 60.7 / 57.14 | 8.1 / 5.75 |
| 0.12 | 0.5 | 33.3 to 48.5 / 48.64 | 4.3 / 3.18 | 66.7 to 97.1 / 97.73 | 6.1 / 5.08 | 41.7 to 60.7 / 57.14 | 8.1 / 5.75 |
| 0.12 | 0.75 | 33.3 to 48.5 / 48.64 | 4.3 / 3.18 | 133.3 to 194.2 / 196.01 | 9.8 / 8.85 | 41.7 to 60.7 / 57.14 | 8.1 / 5.75 |
| 0.14 | 0.25 | 28.6 to 39.8 / 41.73 | 4.0 / 2.89 | 38.1 to 53.1 / 55.67 | 4.5 / 3.46 | 35.7 to 49.8 / 48.88 | 7.6 / 5.20 |
| 0.14 | 0.5 | 28.6 to 39.8 / 41.73 | 4.0 / 2.89 | 57.1 to 79.6 / 83.49 | 5.6 / 4.55 | 35.7 to 49.8 / 48.88 | 7.6 / 5.20 |
| 0.14 | 0.75 | 28.6 to 39.8 / 41.73 | 4.0 / 2.89 | 114.3 to 159.2 / 167.73 | 8.7 / 7.76 | 35.7 to 49.8 / 48.88 | 7.6 / 5.20 |
| 0.16 | 0.25 | 25.0 to 33.5 / 36.56 | 3.8 / 2.67 | 33.3 to 44.6 / 48.70 | 4.3 / 3.18 | 31.2 to 41.8 / 42.80 | 7.2 / 4.78 |
| 0.16 | 0.5 | 25.0 to 33.5 / 36.56 | 3.8 / 2.67 | 50.0 to 66.9 / 73.16 | 5.2 / 4.14 | 31.2 to 41.8 / 42.80 | 7.2 / 4.78 |
| 0.16 | 0.75 | 25.0 to 33.5 / 36.56 | 3.8 / 2.67 | 100.0 to 133.8 / 147.09 | 8.0 / 6.95 | 31.2 to 41.8 / 42.80 | 7.2 / 4.78 |

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

**Net energy per step (`G`).** Same `predicted / measured` format, computed from each row's predicted and measured `T` via `G = M/T − d`:

| `m` | `q` | wander | approach_all | discriminating_wanderer | discriminate | wander_0_5 | approach_all_0_5 |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.25 | −0.488 to −0.475 / −0.483 | −0.130 / −0.106 | −0.483 to −0.463 / −0.474 | +0.081 / +0.101 | −0.491 to −0.480 / −0.485 | −0.305 / −0.285 |
| 0.02 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.488 to −0.475 / −0.483 | −0.094 / −0.095 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 |
| 0.02 | 0.75 | −0.525 to −0.512 / −0.517 | −0.870 / −0.894 | −0.494 to −0.487 / −0.491 | −0.287 / −0.297 | −0.520 to −0.509 / −0.515 | −0.695 / −0.715 |
| 0.04 | 0.25 | −0.474 to −0.450 / −0.466 | +0.128 / +0.218 | −0.460 to −0.425 / −0.448 | +0.520 / +0.632 | −0.479 to −0.460 / −0.470 | −0.169 / −0.113 |
| 0.04 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.501 | −0.474 to −0.450 / −0.466 | +0.240 / +0.286 | −0.500 to −0.500 / −0.500 | −0.500 / −0.501 |
| 0.04 | 0.75 | −0.550 to −0.526 / −0.534 | −1.128 / −1.219 | −0.487 to −0.475 / −0.483 | −0.094 / −0.095 | −0.540 to −0.521 / −0.529 | −0.831 / −0.887 |
| 0.06 | 0.25 | −0.457 to −0.425 / −0.449 | +0.319 / +0.484 | −0.435 to −0.388 / −0.423 | +0.863 / +1.081 | −0.465 to −0.440 / −0.456 | −0.069 / +0.031 |
| 0.06 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.457 to −0.425 / −0.449 | +0.520 / +0.631 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 |
| 0.06 | 0.75 | −0.575 to −0.543 / −0.551 | −1.319 / −1.485 | −0.478 to −0.463 / −0.474 | +0.081 / +0.100 | −0.560 to −0.535 / −0.544 | −0.931 / −1.030 |
| 0.08 | 0.25 | −0.438 to −0.400 / −0.431 | +0.466 / +0.708 | −0.407 to −0.350 / −0.397 | +1.138 / +1.470 | −0.451 to −0.420 / −0.442 | +0.008 / +0.157 |
| 0.08 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.438 to −0.400 / −0.431 | +0.757 / +0.939 | −0.500 to −0.500 / −0.500 | −0.500 / −0.499 |
| 0.08 | 0.75 | −0.600 to −0.562 / −0.569 | −1.466 / −1.709 | −0.469 to −0.450 / −0.466 | +0.240 / +0.287 | −0.580 to −0.549 / −0.559 | −1.008 / −1.157 |
| 0.10 | 0.25 | −0.418 to −0.375 / −0.415 | +0.582 / +0.903 | −0.377 to −0.312 / −0.371 | +1.365 / +1.810 | −0.435 to −0.400 / −0.427 | +0.069 / +0.268 |
| 0.10 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.418 to −0.375 / −0.414 | +0.961 / +1.216 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 |
| 0.10 | 0.75 | −0.625 to −0.582 / −0.586 | −1.582 / −1.904 | −0.459 to −0.438 / −0.457 | +0.386 / +0.464 | −0.600 to −0.565 / −0.573 | −1.069 / −1.268 |
| 0.12 | 0.25 | −0.397 to −0.350 / −0.397 | +0.676 / +1.073 | −0.346 to −0.275 / −0.345 | +1.554 / +2.115 | −0.418 to −0.380 / −0.412 | +0.119 / +0.369 |
| 0.12 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 | −0.397 to −0.350 / −0.397 | +1.138 / +1.470 | −0.500 to −0.500 / −0.500 | −0.500 / −0.501 |
| 0.12 | 0.75 | −0.650 to −0.603 / −0.604 | −1.676 / −2.074 | −0.449 to −0.425 / −0.448 | +0.520 / +0.631 | −0.620 to −0.582 / −0.588 | −1.119 / −1.370 |
| 0.14 | 0.25 | −0.374 to −0.325 / −0.380 | +0.754 / +1.230 | −0.312 to −0.237 / −0.320 | +1.714 / +2.391 | −0.400 to −0.360 / −0.398 | +0.160 / +0.463 |
| 0.14 | 0.5 | −0.500 to −0.500 / −0.501 | −0.500 / −0.497 | −0.374 to −0.325 / −0.380 | +1.294 / +1.700 | −0.500 to −0.500 / −0.500 | −0.500 / −0.500 |
| 0.14 | 0.75 | −0.675 to −0.626 / −0.621 | −1.754 / −2.227 | −0.437 to −0.412 / −0.440 | +0.643 / +0.788 | −0.640 to −0.600 / −0.603 | −1.160 / −1.461 |
| 0.16 | 0.25 | −0.351 to −0.300 / −0.363 | +0.820 / +1.371 | −0.276 to −0.200 / −0.294 | +1.852 / +2.645 | −0.380 to −0.340 / −0.383 | +0.195 / +0.547 |
| 0.16 | 0.5 | −0.500 to −0.500 / −0.500 | −0.500 / −0.499 | −0.351 to −0.300 / −0.363 | +1.431 / +1.914 | −0.500 to −0.500 / −0.501 | −0.500 / −0.500 |
| 0.16 | 0.75 | −0.700 to −0.649 / −0.637 | −1.820 / −2.371 | −0.425 to −0.400 / −0.431 | +0.757 / +0.939 | −0.660 to −0.620 / −0.617 | −1.195 / −1.547 |

</details>

## Stage 2: reproduction calibration

**Point:** find the reproduction threshold `R_thresh` at which a random-like agent (`wander_0_5`) averages about 0.8 offspring per lifetime: low enough that a random starting population declines, high enough that mutation has time (~10 generations) to find something better before it dies out. **Why:** evolution acts on offspring, not energy, and `G` alone can't say how energy converts into births. At the chosen parameters `wander_0_5`'s average lifetime energy drift is about `−200`, but single-meal swings (`+62` edible, `−31` poisonous) are `±500`, so luck dominates whether a given lineage actually reproduces; that has to be measured per-seed, not calculated from `G`. **Gives:** a calibrated `R_thresh`, and confirmation that reproduction counts track `G` for the policies where luck matters less.

Environment (derived from the Stage 1 `T` fits at `m=0.04`): `q=0.5`, `P=0.5`, `N=62`, `d=0.1`, `R_cost=200` (equal to a newborn's starting energy, so reproduction neither creates nor destroys energy). This keeps `wander_0_5` just below break-even (`G≈−0.010`) while clearly rewarding both routes to discrimination: `discriminating_wanderer` (`G≈+0.109`) beats `wander` (`G≈+0.005`), and `discriminate` reproduces about `2.2×` faster than `approach_all` (`G≈+4.71` vs `+2.11`).

| Policy | `G` | Predicted |
|---|---|---|
| `wander_0_5` | −0.010 | drift-only lifespan ≈19,400 steps; offspring per lifetime falls as `R_thresh` rises; wide spread from luck |
| `wander` | +0.005 | barely positive; some die early, some reach the step limit; more offspring than `wander_0_5` at every `R_thresh` |
| `discriminating_wanderer` | +0.109 | mostly survives; ~1 birth every 1,830 steps once above `R_thresh` |
| `approach_all` | +2.11 | ~1 birth every 95 steps |
| `discriminate` | +4.71 | ~1 birth every 42 steps |

`R_thresh` swept over `{250, 300, 350, 400}` (above 200, so the parent survives reproducing), 1,000 seeds each, agent starts at 200 energy, 100,000-step cap with starvation on. Offspring are counted but not added to the world, keeping agent density equal to Stage 1 (so its `T` values stay valid) and isolating the energy-to-offspring question from population dynamics. Measured:

| Policy | `R_thresh` | offspring mean/median/p90 | frac ≥1 | mean lifespan | frac censored | births/1000 steps (censored) |
|---|---|---|---|---|---|---|
| `wander_0_5` | 250 | 0.96 / 1 / 2 | 0.724 | 2,023 | 0 | n/a |
| `wander_0_5` | 300 | 0.92 / 1 / 2 | 0.587 | 3,418 | 0 | n/a |
| `wander_0_5` | 350 | 0.88 / 1 / 2 | 0.502 | 4,579 | 0 | n/a |
| `wander_0_5` | 400 | 0.84 / 0 / 3 | 0.426 | 5,923 | 0 | n/a |
| `wander` | 250 | 1.15 / 1 / 2 | 0.774 | 2,011 | 0 | n/a |
| `wander` | 300 | 1.22 / 1 / 3 | 0.670 | 3,462 | 0 | n/a |
| `wander` | 350 | 1.25 / 1 / 3 | 0.587 | 4,803 | 0 | n/a |
| `wander` | 400 | 1.35 / 1 / 4 | 0.537 | 6,357 | 0 | n/a |
| `discriminating_wanderer` | 250 | 6.87 / 5 / 15 | 0.996 | 10,378 | 0 | n/a |
| `discriminating_wanderer` | 300 | 22.6 / 17 / 55 | 0.995 | 38,363 | 0.094 | 0.60 |
| `discriminating_wanderer` | 350 | 41.9 / 50 / 63 | 0.995 | 73,349 | 0.510 | 0.57 |
| `discriminating_wanderer` | 400 | 50.6 / 55 / 63 | 0.995 | 89,887 | 0.805 | 0.57 |

Both `wander` and `wander_0_5` are fully resolved (`frac censored = 0`) at every `R_thresh` tested: their `G` is close enough to zero that luck decides the outcome before the 100,000-step cap is reached, matching the prediction. `discriminating_wanderer`'s births-per-1,000-steps among censored runs (`≈0.57 to 0.60`) matches its predicted rate almost exactly (`G/R_cost × 1000 = 0.109/200 × 1000 ≈ 0.55`), confirming reproduction counts track `G` once luck stops dominating.

**Chosen: `R_thresh = 400`.** `wander_0_5`'s mean offspring (`0.84`) lands in the target range, `wander` (`1.35`) and `discriminating_wanderer` (`50.6`) clearly exceed it, and `R_cost = 200` stays unchanged.

## Stage 3: evolution (next)

Turn mutation on. First single-founder evolving runs, then full runs from 100 random founders, using eating and approach probes to detect when approach and discrimination evolve, and by which route. A fitness-valley sweep over `P`, `q` and the agent-to-mushroom ratio then tests whether discrimination emerges where the advantage is predicted to be large.

**Settings**

| Setting | Value | Reason |
|---|---|---|
| Environment | `N=62, P=0.5, q=0.5, d=0.1, m=0.04` | same as Stage 2 |
| `R_thresh` / `R_cost` | 400 / 200, given to the newborn | from Stage 2; a birth creates and destroys no energy |
| Offspring | placed on the grid | |
| `A_max` | 1,000 | |
| Grid | 150 × 150 = 22,500 cells, 900 mushrooms | a full population (0.044 agents per cell) stays well below the crowding threshold of about 0.17 agents per cell |

**Cap rule.** When the population is at `A_max` and an agent reaches `R_thresh`, it still reproduces. The newborn replaces a randomly chosen agent (never the parent), so every birth at the cap is also a death and the population keeps turning over.
