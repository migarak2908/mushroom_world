"""
Video replay of the last ~1000 steps of a specific (config, seed, condition)
run: edible vs. poisonous mushrooms distinguishable by color, living agents
colored by their individual eating-discrimination score.

This can't be built from saved sweep data -- sweep.py only logs chunk-level
aggregate counts, not per-step positions. So this re-runs that exact config
+ seed deterministically (same LHS config + same jax.random.key(seed) +
same code path => identical trajectory to the original sweep run, as long
as the code hasn't changed since) and only records positions for the final
window, to avoid holding 100k steps of position arrays in memory for no
reason.

Mechanics:
  1. Run the normal (unrecorded) chunked simulation loop, identical to
     sweep.py/ablation_sweep.py, for every step before the recording window.
  2. Switch to a step function that also emits agent/mushroom positions,
     for just the final WINDOW_STEPS steps.
  3. Probe the population once, at the end of the recorded window, to get
     each living agent's eat_disc score for coloring.
  4. Render sampled frames as a video (mp4 via ffmpeg, falls back to a gif
     via Pillow if ffmpeg isn't found).

Known simplification: agent color is a single end-of-window probe applied
to whichever agent occupies a slot at each frame. If a slot's agent died
and was replaced by a newborn partway through the window, earlier frames
will show the newborn's (not the predecessor's) score. Over a 1000-step
window relative to typical agent lifespans this is usually minor; slots
that are already dead by the end of the window (never resurrected) are
drawn in flat neutral gray in the frames where they were still alive,
since no end-of-window score exists for them.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.stats import qmc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba

from mushroom_world import MushroomWorld
from archive.probe import probe_population

# ---------------------------------------------------------------------------
# Which run to replay
# ---------------------------------------------------------------------------
CONFIG_ID = 9
SEED = 2
RECURRENT = True
PAIN_PLEASURE = True

WINDOW_STEPS = 1000       # how many final steps to record
SAMPLE_EVERY = 2          # use every Nth recorded step as a video frame
FPS = 20
PROBE_REPLICATES_FOR_COLOR = 32
OUT_PATH = f"cfg{CONFIG_ID:03d}_seed{SEED}_replay.mp4"

AGENT_MARKER_SIZE = 25
MUSHROOM_MARKER_SIZE = 90

# ---------------------------------------------------------------------------
# Fixed params -- must match sweep.py / ablation_sweep.py for the trajectory
# to actually reproduce the historical run
# ---------------------------------------------------------------------------
SX = 100
SY = 100
NB_AGENTS = 2000
MAX_AGENTS = 5000
NB_MUSHROOMS = 100
ENERGY_START = 200.0
MUSHROOM_NUTRITION = 20.0
REPROD_THRESHOLD = 210.
PERC_RADIUS = 10
SHUFFLE_PERIOD = 200
NO_SIGNAL = True
H_SIZE = 5
STEPS = 100_000
CHUNK = 100

N_CONFIGS = 100
LHS_SEED = 42

# dataviz reference palette: diverging blue<->red for agent competency
# (blue = discriminates correctly, gray = indifferent, red = poison-preferring),
# violet/orange (validated CVD-safe pair, both distinct from the agent scale)
# for the fixed edible/poison mushroom identity
AGENT_CMAP = LinearSegmentedColormap.from_list("disc_diverge", ["#e34948", "#f0efec", "#2a78d6"])
AGENT_NORM = Normalize(vmin=-1, vmax=1)
UNSCORED_COLOR = "#c3c2b7"
MUSH_EDIBLE_COLOR = "#4a3aa7"
MUSH_POISON_COLOR = "#eb6834"


def generate_configs(n_configs, seed):
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    samples = sampler.random(n=n_configs)
    configs = []
    for i, s in enumerate(samples):
        mutation_std = float(10 ** (np.log10(0.005) + s[0] * (np.log10(0.2) - np.log10(0.005))))
        poison_multiplier = float(-2 - s[1] * 6)
        reprod_cost = float(s[2] * 12)
        energy_decay = float(0.05 + s[3] * 0.10)
        configs.append({
            "config_id": i, "mutation_std": mutation_std, "poison_multiplier": poison_multiplier,
            "reprod_cost": reprod_cost, "energy_decay": energy_decay,
        })
    return configs


def run_and_record(cfg, seed, recurrent, pain_pleasure, window_steps):
    key = jax.random.key(seed)
    env = MushroomWorld(
        seed=seed, grid_x=SX, grid_y=SY,
        nb_agents=NB_AGENTS, max_agents=MAX_AGENTS, nb_mushrooms=NB_MUSHROOMS,
        energy_start=ENERGY_START, energy_decay=cfg["energy_decay"],
        mushroom_nutrition=MUSHROOM_NUTRITION,
        reprod_threshold=REPROD_THRESHOLD, reprod_cost=cfg["reprod_cost"],
        shuffle_period=SHUFFLE_PERIOD,
        mutation_std=cfg["mutation_std"],
        poison_multiplier=cfg["poison_multiplier"],
        frozen_baseline=False,
        no_signal=NO_SIGNAL,
        recurrent=recurrent,
        pain_pleasure=pain_pleasure,
        h_size=H_SIZE,
    )
    agents, mushrooms = env.reset_fn()
    mush_type = np.array(mushrooms.type)  # fixed for the whole run, never reassigned on respawn
    dynamic_agents, static_agents = eqx.partition(agents, eqx.is_array)

    def step_plain(carry, _):
        key, dynamic_agents, mushrooms = carry
        agents = eqx.combine(dynamic_agents, static_agents)
        key, sk1, sk2, sk3, sk4 = jax.random.split(key, 5)
        obs = env._compute_obs(sk1, agents, mushrooms, PERC_RADIUS)
        probs, hidden = eqx.filter_vmap(lambda n, o, h: n(o, h))(agents.network, obs, agents.hidden)
        actions = jax.random.bernoulli(sk2, probs).astype(jnp.int32)
        agents = eqx.tree_at(lambda a: a.hidden, agents, hidden)
        agents, mushrooms, edible, poisonous = env._compute_update(sk3, actions, agents, mushrooms)
        agents = env._compute_reproduce(sk4, agents, env.mutation_std)
        n_alive = agents.alive.sum()
        dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
        return (key, dynamic_agents, mushrooms), n_alive

    def step_record(carry, _):
        key, dynamic_agents, mushrooms = carry
        agents = eqx.combine(dynamic_agents, static_agents)
        key, sk1, sk2, sk3, sk4 = jax.random.split(key, 5)
        obs = env._compute_obs(sk1, agents, mushrooms, PERC_RADIUS)
        probs, hidden = eqx.filter_vmap(lambda n, o, h: n(o, h))(agents.network, obs, agents.hidden)
        actions = jax.random.bernoulli(sk2, probs).astype(jnp.int32)
        agents = eqx.tree_at(lambda a: a.hidden, agents, hidden)
        agents, mushrooms, edible, poisonous = env._compute_update(sk3, actions, agents, mushrooms)
        agents = env._compute_reproduce(sk4, agents, env.mutation_std)
        n_alive = agents.alive.sum()
        dynamic_agents, _ = eqx.partition(agents, eqx.is_array)
        record = (agents.posx, agents.posy, agents.alive, mushrooms.posx, mushrooms.posy)
        return (key, dynamic_agents, mushrooms), (n_alive, record)

    @eqx.filter_jit
    def run_chunk_plain(key, dynamic_agents, mushrooms):
        return jax.lax.scan(step_plain, (key, dynamic_agents, mushrooms), None, length=CHUNK)

    @eqx.filter_jit
    def run_chunk_record(key, dynamic_agents, mushrooms):
        return jax.lax.scan(step_record, (key, dynamic_agents, mushrooms), None, length=CHUNK)

    n_chunks = STEPS // CHUNK
    recording_chunks = max(1, window_steps // CHUNK)
    plain_chunks = n_chunks - recording_chunks

    print(f"Replaying cfg{cfg['config_id']} seed{seed} recurrent={recurrent} pain_pleasure={pain_pleasure}")
    print(f"  {plain_chunks} chunks unrecorded, then {recording_chunks} chunks ({recording_chunks*CHUNK} steps) recorded")

    for c in range(plain_chunks):
        (key, dynamic_agents, mushrooms), n_alive = run_chunk_plain(key, dynamic_agents, mushrooms)
        if int(np.array(n_alive)[-1]) == 0:
            raise RuntimeError(
                f"Population went extinct at chunk {c}, before reaching the recording window "
                f"(window starts at chunk {plain_chunks}). Nothing to replay -- pick a config/seed "
                f"with extinction_rate < 1 for this condition, or shorten WINDOW_STEPS."
            )
        if c % 100 == 0:
            print(f"  chunk {c}/{plain_chunks} (unrecorded), n_alive={int(np.array(n_alive)[-1])}")

    posx_frames, posy_frames, alive_frames = [], [], []
    mush_posx_frames, mush_posy_frames = [], []
    extinct_mid_window = False

    for c in range(recording_chunks):
        (key, dynamic_agents, mushrooms), (n_alive, record) = run_chunk_record(key, dynamic_agents, mushrooms)
        posx, posy, alive, mush_posx, mush_posy = record
        posx_frames.append(np.array(posx))
        posy_frames.append(np.array(posy))
        alive_frames.append(np.array(alive))
        mush_posx_frames.append(np.array(mush_posx))
        mush_posy_frames.append(np.array(mush_posy))
        if int(np.array(n_alive)[-1]) == 0:
            print(f"  population went extinct mid-window at recorded chunk {c}")
            extinct_mid_window = True
            break

    posx_frames = np.concatenate(posx_frames, axis=0)   # (n_steps, MAX_AGENTS)
    posy_frames = np.concatenate(posy_frames, axis=0)
    alive_frames = np.concatenate(alive_frames, axis=0)
    mush_posx_frames = np.concatenate(mush_posx_frames, axis=0)  # (n_steps, NB_MUSHROOMS)
    mush_posy_frames = np.concatenate(mush_posy_frames, axis=0)

    final_agents = eqx.combine(dynamic_agents, static_agents)
    return {
        "posx": posx_frames, "posy": posy_frames, "alive": alive_frames,
        "mush_posx": mush_posx_frames, "mush_posy": mush_posy_frames, "mush_type": mush_type,
        "final_agents": final_agents, "extinct_mid_window": extinct_mid_window,
    }


def probe_final_population(final_agents, n_replicates):
    alive_np = np.array(final_agents.alive).astype(bool)
    n_alive = int(alive_np.sum())
    eat_disc = np.full(MAX_AGENTS, np.nan, dtype=float)
    if n_alive == 0:
        print("  population is extinct at the end of the recorded window -- no probe, agents drawn unscored")
        return eat_disc
    print(f"  probing final population ({n_alive} living agents) at {n_replicates} replicates for coloring...")
    result = probe_population(final_agents.network, jax.random.key(999), n_replicates=n_replicates)
    eat_disc[alive_np] = np.asarray(result["eat_disc"])[alive_np]
    return eat_disc


def make_video(data, eat_disc_by_slot, sample_every, fps, out_path):
    n_steps = data["alive"].shape[0]
    frame_indices = np.arange(0, n_steps, sample_every)

    mush_colors = np.where(data["mush_type"] == 1, MUSH_EDIBLE_COLOR, MUSH_POISON_COLOR)

    agent_rgba = AGENT_CMAP(AGENT_NORM(np.nan_to_num(eat_disc_by_slot, nan=0.0)))
    unscored = np.isnan(eat_disc_by_slot)
    agent_rgba[unscored] = to_rgba(UNSCORED_COLOR)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_xlim(-1, SX + 1)
    ax.set_ylim(-1, SY + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    mush_scat = ax.scatter([], [], s=MUSHROOM_MARKER_SIZE, marker="o", edgecolors="none", zorder=2)
    agent_scat = ax.scatter([], [], s=AGENT_MARKER_SIZE, marker="o", edgecolors="none", zorder=3)
    title = ax.set_title("", fontsize=10)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=MUSH_EDIBLE_COLOR, markersize=9, label="edible mushroom"),
        plt.Line2D([], [], marker="o", linestyle="none", color=MUSH_POISON_COLOR, markersize=9, label="poisonous mushroom"),
        plt.Line2D([], [], marker="o", linestyle="none", color="#2a78d6", markersize=7, label="agent: discriminates well"),
        plt.Line2D([], [], marker="o", linestyle="none", color="#e34948", markersize=7, label="agent: poison-preferring"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=8, frameon=False)

    def update(frame_num):
        step = frame_indices[frame_num]
        mush_scat.set_offsets(np.column_stack([data["mush_posx"][step], data["mush_posy"][step]]))
        mush_scat.set_color(mush_colors)

        alive_mask = data["alive"][step].astype(bool)
        agent_scat.set_offsets(np.column_stack([data["posx"][step][alive_mask], data["posy"][step][alive_mask]]))
        agent_scat.set_color(agent_rgba[alive_mask])

        title.set_text(f"cfg{CONFIG_ID} seed{SEED}  --  step {step} of recorded window  ({int(alive_mask.sum())} alive)")
        return mush_scat, agent_scat, title

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer, dpi=140)
        print(f"Saved {out_path}")
    except FileNotFoundError:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        print(f"ffmpeg not found -- falling back to gif: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=110)
        print(f"Saved {gif_path}")

    plt.close(fig)


def main():
    all_configs = {c["config_id"]: c for c in generate_configs(N_CONFIGS, LHS_SEED)}
    cfg = all_configs[CONFIG_ID]

    data = run_and_record(cfg, SEED, RECURRENT, PAIN_PLEASURE, WINDOW_STEPS)
    eat_disc_by_slot = probe_final_population(data["final_agents"], PROBE_REPLICATES_FOR_COLOR)
    make_video(data, eat_disc_by_slot, SAMPLE_EVERY, FPS, OUT_PATH)


if __name__ == "__main__":
    main()
