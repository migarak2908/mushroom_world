import jax.numpy as jnp
import pandas as pd
import wandb


def compute_metrics(eat_deltas, num_steps, energy_decay):
    """Reduce a (num_seeds, num_steps, nb_agents) eat_delta history to T/T_first/fraction_poisonous/G per seed."""
    meals = eat_deltas != 0
    meal_count = meals.sum(axis=(1, 2))
    T = jnp.where(meal_count > 0, num_steps / meal_count, jnp.nan)

    meal_occurred = meals.any(axis=2)
    first_meal_idx = jnp.argmax(meal_occurred.astype(jnp.int32), axis=1)
    T_first = jnp.where(meal_count > 0, (first_meal_idx + 1).astype(jnp.float32), jnp.nan)

    fraction_poisonous = jnp.where(meal_count > 0, (eat_deltas < 0).sum(axis=(1, 2)) / meal_count, jnp.nan)
    G = eat_deltas.mean(axis=(1, 2)) - energy_decay
    return T, T_first, fraction_poisonous, G


def compute_reproduction_metrics(births, alive, eat_deltas, num_steps):
    """Reduce (num_seeds, num_steps, nb_agents) birth/alive/eat_delta histories to per-seed reproduction stats."""
    offspring_count = births.sum(axis=(1, 2))

    alive_bool = alive.astype(bool)
    censored = alive_bool[:, -1, :].any(axis=1)

    dead_occurred = (~alive_bool).any(axis=2)
    any_dead = dead_occurred.any(axis=1)
    first_dead_idx = jnp.argmax(dead_occurred.astype(jnp.int32), axis=1)
    lifespan = jnp.where(any_dead, first_dead_idx, num_steps).astype(jnp.float32)

    birth_occurred = births.any(axis=2)
    any_birth = birth_occurred.any(axis=1)
    first_birth_idx = jnp.argmax(birth_occurred.astype(jnp.int32), axis=1)
    first_birth_step = jnp.where(any_birth, (first_birth_idx + 1).astype(jnp.float32), jnp.nan)

    meals_edible = (eat_deltas > 0).sum(axis=(1, 2))
    meals_poisonous = (eat_deltas < 0).sum(axis=(1, 2))

    return dict(
        offspring_count=offspring_count,
        censored=censored,
        lifespan=lifespan,
        first_birth_step=first_birth_step,
        meals_edible=meals_edible,
        meals_poisonous=meals_poisonous,
    )


def summarize_condition(metrics):
    """Reduce compute_reproduction_metrics' per-seed dict to one row per (policy, R_thresh) condition."""
    offspring_count = metrics["offspring_count"]
    lifespan = metrics["lifespan"]
    censored = metrics["censored"]

    censored_offspring = offspring_count[censored]
    censored_lifespan = lifespan[censored]
    births_per_1000_censored = (
        float((censored_offspring / censored_lifespan * 1000).mean())
        if censored_lifespan.size > 0 else float("nan")
    )

    return dict(
        offspring_mean=float(offspring_count.mean()),
        offspring_median=float(jnp.median(offspring_count)),
        offspring_frac_at_least_1=float((offspring_count >= 1).mean()),
        offspring_p90=float(jnp.percentile(offspring_count, 90)),
        lifespan_mean=float(lifespan.mean()),
        censored_frac=float(censored.mean()),
        births_per_1000_censored=births_per_1000_censored,
    )


def log_results(policy_name, config, T, T_first, fraction_poisonous, G):
    wandb.init(project="mushroom-language", config={**config, "policy": policy_name})

    n_zero_meal_seeds = int(jnp.isnan(T).sum())
    T_valid = T[~jnp.isnan(T)]
    T_first_valid = T_first[~jnp.isnan(T_first)]

    summary = dict(
        T_mean=float(jnp.nanmean(T)), T_std=float(jnp.nanstd(T)),
        T_first_mean=float(jnp.nanmean(T_first)), T_first_std=float(jnp.nanstd(T_first)),
        fraction_poisonous_mean=float(jnp.nanmean(fraction_poisonous)),
        G_mean=float(G.mean()), G_std=float(G.std()),
        zero_meal_seeds=n_zero_meal_seeds,
    )

    wandb.log({
        **summary,
        "T_histogram": wandb.Histogram(T_valid) if T_valid.size > 0 else None,
        "T_first_histogram": wandb.Histogram(T_first_valid) if T_first_valid.size > 0 else None,
        "G_histogram": wandb.Histogram(G),
        "per_seed": wandb.Table(dataframe=pd.DataFrame({
            "seed": range(len(T)),
            "T": T, "T_first": T_first, "fraction_poisonous": fraction_poisonous, "G": G,
        })),
    })
    wandb.finish()

    return summary
