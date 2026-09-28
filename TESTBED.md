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
| `wander` policy instance | `policy.py` | Done, verified |
| Generic runner (`build_world`, `run_policy`, `run_policy_batch`, `log_results`) | `runner.py` | Done, verified |
| Thin per-policy entry script | `scripts/run_wander.py` | Done, verified |

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
- **Time between meals (wander, eat always on):** `T` should fall between `4/m` (best case, never revisiting a cell) and `4×t*`, where `t*` solves `π·t* / ln(8·t*) = 1/m` (the random-walk cover-time figure — note the doc's upper bound is *4 times* this `t*`, not `t*` itself)
- **Net energy per step:** `G = M / T − d`
- **Fraction poisonous meals:** should track `q` directly, since wander eats indiscriminately

Where `m = nb_mushrooms / (SX × SY)` is mushroom density. Per the doc, `G` for any `(N, P)` is recoverable after the fact from the meal log without rerunning: `run_policy` logs realised energy deltas, but the *sign* of each nonzero entry (`>0` edible, `<0` poison) is all that's needed to recompute `G` for a different `(N', P')` — the magnitude doesn't need to be "undone" first.

### Sweep design (Stage 1, §4 of the design doc)

Fixed for every run in the sweep:

| Symbol | Meaning | Value |
|---|---|---|
| `SX`, `SY` | Grid size | 20, 20 |
| `N` (`mushroom_nutrition`) | Energy per edible meal | 10.0 |
| `P` (`poison_multiplier`) | Poison energy penalty multiplier | 1.0 |
| `d` (`energy_decay`) | Energy lost per step | 0.5 |
| `energy_start` | Starting energy | 200.0 |
| `death_enabled` | | `False` (Stage 1 requirement) |
| Number of steps per seed | | 20,000 |
| Number of seeds per config | | 1,000 |

Swept: `m ∈ {0.01, 0.02, 0.04, 0.08}` (→ `nb_mushrooms ∈ {4, 8, 16, 32}` at this grid size) `× q ∈ {0.25, 0.5, 0.75}` — 12 configs.

### Predicted values, computed from the formulas above with the fixed parameters

| `m` (`nb_mushrooms`) | `q` | `T` predicted range | `M_all` | `G` predicted range |
|---|---|---|---|---|
| 0.01 (4) | 0.25 | [400, 963] | 5.0 | [−0.495, −0.487] |
| 0.01 (4) | 0.5 | [400, 963] | 0.0 | −0.500 |
| 0.01 (4) | 0.75 | [400, 963] | −5.0 | [−0.512, −0.505] |
| 0.02 (8) | 0.25 | [200, 430] | 5.0 | [−0.488, −0.475] |
| 0.02 (8) | 0.5 | [200, 430] | 0.0 | −0.500 |
| 0.02 (8) | 0.75 | [200, 430] | −5.0 | [−0.525, −0.512] |
| 0.04 (16) | 0.25 | [100, 189] | 5.0 | [−0.474, −0.450] |
| 0.04 (16) | 0.5 | [100, 189] | 0.0 | −0.500 |
| 0.04 (16) | 0.75 | [100, 189] | −5.0 | [−0.550, −0.526] |
| 0.08 (32) | 0.25 | [50, 81] | 5.0 | [−0.438, −0.400] |
| 0.08 (32) | 0.5 | [50, 81] | 0.0 | −0.500 |
| 0.08 (32) | 0.75 | [50, 81] | −5.0 | [−0.600, −0.562] |

`fraction poisonous` predicted ≈ `q` in every row (not repeated in the table). At `q=0.5`, `M_all=0` exactly for this `N`/`P`, so `G` collapses to exactly `−d` regardless of `T` — a useful sanity check, and it's exactly what the earlier single-config validation run measured (`G≈-0.50` at `q≈0.5`).

### Results

| `m` | `q` | `T` measured | Fraction poisonous measured | `G` measured | Zero-meal seeds | Match? |
|---|---|---|---|---|---|---|
| 0.01 | 0.25 | 597.21 | 0.256 | −0.4915 | 0/1000 | ✅ |
| 0.01 | 0.5 | 597.21 | 0.502 | −0.5000 | 0/1000 | ✅ |
| 0.01 | 0.75 | 597.21 | 0.747 | −0.5085 | 0/1000 | ✅ |
| 0.02 | 0.25 | 295.85 | 0.256 | −0.4832 | 0/1000 | ✅ |
| 0.02 | 0.5 | 295.85 | 0.505 | −0.5004 | 0/1000 | ✅ |
| 0.02 | 0.75 | 295.85 | 0.749 | −0.5171 | 0/1000 | ✅ |
| 0.04 | 0.25 | 146.90 | 0.253 | −0.4660 | 0/1000 | ✅ |
| 0.04 | 0.5 | 146.90 | 0.501 | −0.5002 | 0/1000 | ✅ |
| 0.04 | 0.75 | 146.90 | 0.747 | −0.5339 | 0/1000 | ✅ |
| 0.08 | 0.25 | 73.22 | 0.253 | −0.4322 | 0/1000 | ✅ |
| 0.08 | 0.5 | 73.22 | 0.498 | −0.4994 | 0/1000 | ✅ |
| 0.08 | 0.75 | 73.22 | 0.745 | −0.5672 | 0/1000 | ✅ |

All 12 configs ran on Colab (GPU), 20,000 steps × 1,000 seeds each, logged to wandb project `mushroom-language`. Every `T` lands inside its predicted `[4/m, 4t*]` range, `fraction_poisonous` tracks `q` to within ~0.01 in every row, and `G` falls inside (or within ~0.003 of) its predicted range everywhere — including the `q=0.5 → G≈−d` sanity check holding at all four densities. No zero-meal seeds anywhere, so no reliability concerns from insufficient steps even at the sparsest density.

### Notes / anomalies

No anomalies — `wander`'s measured behaviour matches the design doc's §4 closed-form predictions across the full `m×q` grid. This validates the environment mechanics (movement, eating, respawn, density) together with the analytical model, and gives a clean baseline to compare `approach-all`/`discriminate` against once they're built. Two real bugs were caught and fixed *during* this validation work (not present in the final numbers above): a broadcasting bug in the overlap check that would have silently mis-attributed which mushroom an agent was standing on, and a zero-meal-seed edge case (`T=inf`/`nan`) at low density that would have corrupted the batch mean if left unhandled.
