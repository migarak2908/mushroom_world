# Modular Foraging Testbed — Ongoing Work

This documents the **new, modular rebuild** of the mushroom-foraging environment (`world.py`, `food_source.py`, `mushrooms.py`, `policy.py`, `observation.py`), which supersedes the exploratory codebase archived at `archive/mushroom_world.py`. See `README.md` for the published ALIFE 2026 poster (LB135) that came out of the old codebase — this file tracks the work that comes after it.

## Why rebuild

The original goal (from the design roadmap): a **reusable, modular testbed** for open-ended, non-episodic evolution of foraging agents, built to answer:

> Under what environment parameters does feature-based discrimination (eating edible mushrooms, avoiding poisonous ones) evolve from randomly initialised agents, in a population where competition keeps it turning over?

The approach: derive simple analytical predictions (an energy budget), validate them against **hand-coded agents** before any evolution runs, then use the validated predictions to choose parameters for evolutionary runs. The rebuild makes the environment's components (food source, respawn policy, observation channels, movement/eat rules) independently swappable, so later stages don't require rewriting the environment.

## What's built so far

| Component | File | Status |
|---|---|---|
| `Agents`/`World` — grid, movement, energy decay, death, eat | `world.py` | Done, verified |
| `FoodSource`/`RespawnStrategy` protocols | `food_source.py` | Done, verified |
| `Mushrooms`/`MushroomSource` — reset, consume + collision-free respawn, `type_at` | `mushrooms.py` | Done, verified |
| `Policy`/`MovementRule`/`EatRule` protocols, `ComposedPolicy` | `policy.py` | Done, verified |
| `RandomMovement`, `AlwaysEat` (→ `wander` baseline) | `policy.py` | Done, verified |
| Rotating 3×5 observation window (`_window_cells`) | `world.py` | Done, verified |
| `ObservationChannel`/`FoodTypeChannel`, `_build_obs` | `observation.py`, `world.py` | Done, verified |
| `wander` policy instance | `policy.py` | Not yet added |
| Outer driving loop (obs → policy → step, over many steps) | *(new script, TBD)* | Not yet built |

## Stage 1 (first test): validating `wander`

Per the roadmap's Stage 1 ("single agent, reproduction off"): run the `wander` hand-coded policy (random movement each step, eat whatever it lands on) and check whether its measured behaviour matches the closed-form predictions derived analytically, **before** trusting anything built on top of it (other baselines, evolution).

### What we're measuring

| Quantity | Meaning |
|---|---|
| `T` | Average number of steps between meals |
| Fraction poisonous | Share of meals that were poisonous mushrooms |
| `G` | Net energy change per step |

### What we're predicting (closed-form, from the design doc §4)

- **Meal value (indiscriminate eater):** `M_all = N × [(1 − q) − q × P]`
- **Time between meals (wander, eat always on):** `T` should fall between `4 / m` (best case, never revisiting a cell) and the random-walk cover-time bound, i.e. the `t` solving `π·t / ln(8t) = 1/m`
- **Net energy per step:** `G = M / T − d`
- **Fraction poisonous meals:** should track `q` directly, since wander eats indiscriminately

Where `m = nb_mushrooms / (SX × SY)` is mushroom density.

### Parameters used for this run

| Symbol | Meaning | Value |
|---|---|---|
| `SX`, `SY` | Grid size | |
| `nb_mushrooms` | Mushroom count | |
| `m` | Density (`nb_mushrooms / (SX×SY)`) | |
| `q` (`poison_proportion`) | Fraction poisonous | |
| `N` (`mushroom_nutrition`) | Energy per edible meal | |
| `P` (`poison_multiplier`) | Poison energy penalty multiplier | |
| `d` (`energy_decay`) | Energy lost per step | |
| `energy_start` | Starting energy | |
| Number of steps run | | |
| Seed(s) | | |

*(fill in once the run is configured)*

### Results

| Quantity | Predicted | Measured | Match? |
|---|---|---|---|
| `T` (range) | | | |
| Fraction poisonous | ≈ `q` | | |
| `G` | | | |

*(fill in once the run completes)*

### Notes / anomalies

*(anything that didn't match the prediction, and why — e.g. bug found, or the analytical approximation breaking down at this density)*
