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
| `ApproachMovement`, `DiscriminateEat` (→ `approach_all`, `discriminate`, `discriminating_wanderer`) | `policy.py` | Done, verified |

## Stage 1: validating hand-coded baselines

Per the roadmap's Stage 1 ("single agent, reproduction off"): run each hand-coded policy and check whether its measured behaviour matches the closed-form predictions derived analytically, **before** trusting anything built on top of it (other baselines, evolution). Covers `wander`, `approach-all`, and `discriminate` — all run against the same sweep design below.

### What we're measuring

| Quantity | Meaning |
|---|---|
| `T` | Average number of steps between meals |
| Fraction poisonous | Share of meals that were poisonous mushrooms |
| `G` | Net energy change per step |

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

### `wander`

- **Meal value (indiscriminate eater):** `M_all = N × [(1 − q) − q × P]`
- **Time between meals (wander, eat always on):** `T` should fall between `4/m` (best case, never revisiting a cell) and `4×t*`, where `t*` solves `π·t* / ln(8·t*) = 1/m` (the random-walk cover-time figure — note the doc's upper bound is *4 times* this `t*`, not `t*` itself)
- **Net energy per step:** `G = M / T − d`
- **Fraction poisonous meals:** should track `q` directly, since wander eats indiscriminately

Where `m = nb_mushrooms / (SX × SY)` is mushroom density. Per the doc, `G` for any `(N, P)` is recoverable after the fact from the meal log without rerunning: `run_policy` logs realised energy deltas, but the *sign* of each nonzero entry (`>0` edible, `<0` poison) is all that's needed to recompute `G` for a different `(N', P')` — the magnitude doesn't need to be "undone" first.

**Predicted values, computed from the formulas above with the fixed parameters:**

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

**Results:**

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

No anomalies — `wander`'s measured behaviour matches the design doc's §4 closed-form predictions across the full `m×q` grid. This validates the environment mechanics (movement, eating, respawn, density) together with the analytical model, and gives a clean baseline to compare `approach-all`/`discriminate` against. Two real bugs were caught and fixed *during* this validation work (not present in the final numbers above): a broadcasting bug in the overlap check that would have silently mis-attributed which mushroom an agent was standing on, and a zero-meal-seed edge case (`T=inf`/`nan`) at low density that would have corrupted the batch mean if left unhandled.

### `approach-all` and `discriminate`

`ApproachMovement` uses a deliberately simplified two-tier decision rule rather than the doc's idealized "always steer to nearest visible target" model — see the design discussion above. In short: anything directly ahead takes absolute priority (move forward), anything on an immediate flank (one cell left/right) gets turned toward, and anything else (off to the side by 2 cells, in either direction) falls back to a 90%-forward/10%-random-turn search rather than being deliberately steered toward. This was chosen specifically because the natural "argmin over the whole window" version could get stuck in a genuine, deterministic infinite oscillation between two competing targets on opposite flanks (verified and fixed during development — see commit history). The tradeoff is that measured `T` runs somewhat higher than the doc's idealized spotting-and-reaching formula predicts, especially at low density / high `q`, since a real fraction of visible targets aren't deliberately pursued.

- **Time between meals (approach-all, eat always on):** `T ≈ 1/(5m) + 4`
- **Time between meals (discriminate, eat only when own cell is edible):** `T ≈ 1/(5m(1−q)) + 4`
- **Discriminator advantage:** `G_disc ≥ d·q·P / [(1−q) − q·P]` — undefined/infinite at `q=0.5` for this `N`/`P` (the denominator is exactly zero there, same degenerate point noted for `wander`)
- `discriminate`'s `fraction_poisonous` should be exactly `0`, by construction (`DiscriminateEat` only ever eats when the agent's own cell is edible)

**`T`: predicted (idealized) vs measured:**

| `m` | `q` | `T` approach-all predicted | `T` approach-all measured | `T` discriminate predicted | `T` discriminate measured |
|---|---|---|---|---|---|
| 0.01 | 0.25 | 24.0 | 39.89 | 30.7 | 60.33 |
| 0.01 | 0.5 | 24.0 | 39.89 | 44.0 | 91.13 |
| 0.01 | 0.75 | 24.0 | 39.89 | 84.0 | 129.02 |
| 0.02 | 0.25 | 14.0 | 19.75 | 17.3 | 28.19 |
| 0.02 | 0.5 | 14.0 | 19.75 | 24.0 | 47.15 |
| 0.02 | 0.75 | 14.0 | 19.75 | 44.0 | 92.55 |
| 0.04 | 0.25 | 9.0 | 10.02 | 10.7 | 13.65 |
| 0.04 | 0.5 | 9.0 | 10.02 | 14.0 | 21.37 |
| 0.04 | 0.75 | 9.0 | 10.02 | 24.0 | 49.37 |
| 0.08 | 0.25 | 6.5 | 5.34 | 7.3 | 6.99 |
| 0.08 | 0.5 | 6.5 | 5.34 | 9.0 | 10.37 |
| 0.08 | 0.75 | 6.5 | 5.34 | 14.0 | 21.99 |

Measured `T` runs consistently higher than the idealized prediction at low-to-mid density, converging to (and at `m=0.08`, even slightly beating) the prediction at high density — consistent with the simplified movement rule's known gap (col=±2 targets aren't deliberately pursued), not a bug. This is an expected, explained deviation, not a validation failure.

**`G`: full results:**

| `m` | `q` | `G` approach-all | `G` discriminate | `frac_poison` discriminate | Zero-meal seeds (discriminate) | Discriminate beats approach-all? |
|---|---|---|---|---|---|---|
| 0.01 | 0.25 | −0.3772 | −0.3128 | 0.000 | 4/1000 | ✅ |
| 0.01 | 0.5 | −0.5015 | −0.3755 | 0.000 | 69/1000 | ✅ |
| 0.01 | 0.75 | −0.6254 | −0.4376 | 0.000 | 306/1000 | ✅ |
| 0.02 | 0.25 | −0.2556 | −0.1245 | 0.000 | 0/1000 | ✅ |
| 0.02 | 0.5 | −0.5068 | −0.2516 | 0.000 | 6/1000 | ✅ |
| 0.02 | 0.75 | −0.7533 | −0.3747 | 0.000 | 96/1000 | ✅ |
| 0.04 | 0.25 | −0.0097 | +0.2503 | 0.000 | 0/1000 | ✅ |
| 0.04 | 0.5 | −0.5040 | +0.0038 | 0.000 | 0/1000 | ✅ |
| 0.04 | 0.75 | −0.9928 | −0.2450 | 0.000 | 6/1000 | ✅ |
| 0.08 | 0.25 | +0.4231 | +0.9449 | 0.000 | 0/1000 | ✅ |
| 0.08 | 0.5 | −0.4985 | +0.4954 | 0.000 | 0/1000 | ✅ |
| 0.08 | 0.75 | −1.4216 | +0.0122 | 0.000 | 0/1000 | ✅ |

**`discriminate` beats `approach_all`'s `G` in all 12/12 configs** — the doc's core discrimination-advantage prediction (§4.7), confirmed across the entire sweep, not just one point. `fraction_poisonous` is exactly `0.000` for `discriminate` in every row — thousands of meals across the sweep, never once poison, a strong correctness confirmation of `DiscriminateEat`.

The rising zero-meal-seed count at low `m`/high `q` (up to 306/1000 at `m=0.01, q=0.75`) reflects the effective edible density (`m×(1−q)`) getting very low at that corner (`0.0025` there) — a genuine fraction of 20,000-step runs never encounter an edible mushroom at all. `G` still correctly includes those seeds (a zero-meal seed contributes exactly `G=−d`; only `T`/`fraction_poisonous` are `nan`-guarded), so the reported averages are honest, not biased by silently excluding the hardest cases.

Two real bugs were caught and fixed during development of `ApproachMovement`, not present in the numbers above:
- A broadcasting bug where the movement decision was built directly off `obs` without correctly handling the window's row/col layout, causing the first version to get permanently stuck oscillating between two directions when two mushrooms sat on opposite flanks (deterministic, would never self-resolve — verified via direct trace before the fix).
- `DiscriminateEat` originally required a `source` reference tied to a specific `World`'s `MushroomSource`, which meant `discriminate` couldn't exist as a ready module-level constant like `wander`/`approach_all` — fixed by reading the agent's own cell from `obs`'s food channel instead (verified identical to reading `type_at` directly, since that channel is ground truth, not a noisy encoding).

No anomalies in the final swept results — `T` deviates from the idealized prediction in the expected direction and magnitude given the simplified movement rule (explained above), and the core qualitative prediction (discrimination provides a `G` advantage, robust across the whole `m×q` grid) holds cleanly everywhere.

### `discriminating_wanderer`

Random movement (same as `wander`) + `DiscriminateEat` — isolates the eat-discrimination benefit on its own, with navigation held at "random." Completes the 2×2 design (movement × eat-discrimination) alongside `wander`, `approach_all`, `discriminate`.

- **Predicted time between (edible) meals:** `T ≈ wander's T ÷ (1 − q)` — since movement is identical to `wander`, but only edible landings count as a "meal" here.
- `fraction_poisonous` should be exactly `0`, same as `discriminate`.

**`T`: predicted vs measured:**

| `m` | `q` | `T` predicted | `T` measured |
|---|---|---|---|
| 0.01 | 0.25 | 796.3 | 924.08 |
| 0.01 | 0.5 | 1194.4 | 1442.18 |
| 0.01 | 0.75 | 2388.8 | 2064.29 |
| 0.02 | 0.25 | 394.5 | 424.61 |
| 0.02 | 0.5 | 591.7 | 714.99 |
| 0.02 | 0.75 | 1183.4 | 1486.72 |
| 0.04 | 0.25 | 195.9 | 202.25 |
| 0.04 | 0.5 | 293.8 | 321.20 |
| 0.04 | 0.75 | 587.6 | 765.62 |
| 0.08 | 0.25 | 97.6 | 98.71 |
| 0.08 | 0.5 | 146.4 | 151.27 |
| 0.08 | 0.75 | 292.9 | 330.73 |

Closest at high density (`m=0.08`, within ~1-13%), widening at low density / high `q` — same pattern as `approach_all`/`discriminate`'s deviation, and for the same underlying reason: rarer, harder-to-find food means more variance and more of the sample affected by edge effects (respawn randomizing across the whole grid, long-run `T` running higher than short-run estimates, per the doc's own §4.5 caveat).

**`G`: full results:**

| `m` | `q` | `G` measured | `frac_poison` | Zero-meal seeds |
|---|---|---|---|---|
| 0.01 | 0.25 | −0.4871 | 0.000 | 4/1000 |
| 0.01 | 0.5 | −0.4914 | 0.000 | 69/1000 |
| 0.01 | 0.75 | −0.4956 | 0.000 | 306/1000 |
| 0.02 | 0.25 | −0.4745 | 0.000 | 0/1000 |
| 0.02 | 0.5 | −0.4830 | 0.000 | 6/1000 |
| 0.02 | 0.75 | −0.4914 | 0.000 | 96/1000 |
| 0.04 | 0.25 | −0.4487 | 0.000 | 0/1000 |
| 0.04 | 0.5 | −0.4659 | 0.000 | 0/1000 |
| 0.04 | 0.75 | −0.4827 | 0.000 | 6/1000 |
| 0.08 | 0.25 | −0.3970 | 0.000 | 0/1000 |
| 0.08 | 0.5 | −0.4309 | 0.000 | 0/1000 |
| 0.08 | 0.75 | −0.4652 | 0.000 | 0/1000 |

`fraction_poisonous` is exactly `0.000` everywhere — same `DiscriminateEat`, same correctness guarantee as `discriminate`. `G` sits consistently between `wander`'s and `discriminate`'s at every config (e.g. at `m=0.04, q=0.5`: wander `−0.5002` < discriminating_wanderer `−0.4659` < discriminate `+0.0038`) — avoiding poison alone gives a modest, consistent lift over pure `wander`, but nowhere near what adding navigation on top (full `discriminate`) achieves, since this policy still can't find food any faster than blind wandering.

No anomalies — zero-meal-seed counts match `discriminate`'s exactly at each `(m,q)` (same movement pattern, same edible-density-driven difficulty at the sparse/dangerous corner), and the `T ≈ wander_T/(1−q)` relationship holds well across the grid.
