"""Benchmark probe_population at different replicate counts on a saved agent file.

Run this before committing to a PROBE_REPLICATES value for the full sweep --
whether cost scales sub-linearly (batch grows into unused GPU parallelism) or
~linearly (single dispatch already saturates the device) can only be measured,
not reasoned about in the abstract.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import jax

from archive.probe import probe_population, load_networks, MAX_AGENTS
from agent import Network
import equinox as eqx

AGENTS_DIR = "/content/drive/MyDrive/mushroom_sweep_probed_agents_nosig"
REPLICATE_COUNTS = [8, 32, 100]


def find_one_agent_file(agents_dir):
    if not os.path.isdir(agents_dir):
        return None
    for fname in sorted(os.listdir(agents_dir)):
        if fname.endswith("_agents.eqx"):
            return fname
    return None


def make_dummy_networks(seed=0, h_size=5):
    keys = jax.random.split(jax.random.key(seed), MAX_AGENTS)
    return eqx.filter_vmap(lambda k: Network(k, input_dim=17, h_size=h_size, output_dim=5))(keys)


def main():
    fname = find_one_agent_file(AGENTS_DIR)

    if fname is None:
        print(f"No saved agent file found in {AGENTS_DIR} -- using a freshly initialised "
              f"random population instead. Timings will still show the compile vs. steady-state "
              f"and scaling-with-N pattern, but won't reflect a trained/evolved network's behavior.")
        networks = make_dummy_networks()
    else:
        base = fname.replace("_agents.eqx", "")
        alive_path = os.path.join(AGENTS_DIR, f"{base}_alive.npy")
        print(f"Using saved agent file: {fname}")
        networks, _ = load_networks(os.path.join(AGENTS_DIR, fname), alive_path)

    print(f"{'N':>5}  {'first call (s)':>16}  {'second call (s)':>17}  {'per-agent-replicate (ms)':>26}")

    for n_replicates in REPLICATE_COUNTS:
        key = jax.random.key(n_replicates)

        try:
            # probe_population converts every result to numpy internally, which already
            # forces a device sync -- no extra block_until_ready needed here.
            t0 = time.perf_counter()
            probe_population(networks, key, n_replicates=n_replicates)
            t1 = time.perf_counter()

            key, subkey = jax.random.split(key)
            probe_population(networks, subkey, n_replicates=n_replicates)
            t2 = time.perf_counter()

            first_call = t1 - t0
            second_call = t2 - t1
            per_unit_ms = 1000 * second_call / (MAX_AGENTS * n_replicates)

            print(f"{n_replicates:>5}  {first_call:>16.2f}  {second_call:>17.2f}  {per_unit_ms:>26.5f}")

        except Exception as e:
            print(f"{n_replicates:>5}  FAILED: {type(e).__name__}: {e}")
            print("  (if this is an OOM, that's the signal that cost goes ~linear past this N "
                  "on this hardware -- consider chunking agents or lowering N)")


if __name__ == "__main__":
    main()
