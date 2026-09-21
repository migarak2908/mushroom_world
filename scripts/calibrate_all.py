"""Run the 8-condition calibration sweep (5 seeds each, 40 runs) and write
one summary CSV, then report the pass/fail checks from the calibration
spec. Uses controllers.build_policy's default epsilon=0.05, k=5 (discriminate
now needs both to avoid the deterministic-cycling / avoidance-barrier
failure modes documented in controllers.py).

Each condition answers a question no other condition answers:
  1  wander        n=1     m=1000  p=-1.0  floor: chance encounter rate; slope ~= -decay
  2  approach_all  n=1     m=1000  p=-1.0  tau without crowding; slope should equal -decay exactly
  3  approach_all  n=1     m=1000  p=-0.5  slope flips positive: confirms P=-1 boundary
  4  discriminate  n=1     m=1000  p=-1.0  tau_e and kappa = tau_e / tau
  5  approach_all  n=500   m=1000  p=-1.0  tau at 0.5 agents per mushroom
  6  approach_all  n=2000  m=1000  p=-1.0  tau at 2 agents per mushroom
  7  discriminate  n=500   m=1000  p=-1.0  kappa at 0.5 per mushroom
  8  discriminate  n=2000  m=1000  p=-1.0  kappa at 2 per mushroom
  9  approach_all  n=1     m=100   p=-1.0  (optional) comparison with the poster world
"""
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibrate import out_path
from calibrate_analysis import analyze_run

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATE_PY = os.path.join(SCRIPT_DIR, "calibrate.py")

SEEDS = range(5)
OUT_DIR = "./results/calibration/"
SUMMARY_PATH = os.path.join(OUT_DIR, "calibration_summary.csv")

CONDITIONS = [
    dict(id=1, strategy="wander", n_agents=1, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=2, strategy="approach_all", n_agents=1, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=3, strategy="approach_all", n_agents=1, nb_mushrooms=1000, poison_multiplier=-0.5),
    dict(id=4, strategy="discriminate", n_agents=1, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=5, strategy="approach_all", n_agents=500, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=6, strategy="approach_all", n_agents=2000, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=7, strategy="discriminate", n_agents=500, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=8, strategy="discriminate", n_agents=2000, nb_mushrooms=1000, poison_multiplier=-1.0),
    dict(id=9, strategy="approach_all", n_agents=1, nb_mushrooms=100, poison_multiplier=-1.0),
]

COMMON = dict(grid_size=100, nutrition=20.0, energy_decay=0.1, energy_start=10000.0,
              perc_radius=10, steps=10000, epsilon=0.05, k=5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for cond in CONDITIONS:
        for seed in SEEDS:
            path = out_path(OUT_DIR, cond["strategy"], cond["n_agents"], cond["nb_mushrooms"],
                             cond["poison_multiplier"], COMMON["epsilon"], COMMON["k"], seed)
            if os.path.exists(path):
                row = analyze_run(path)
                row["condition_id"] = cond["id"]
                rows.append(row)
                print(f"[cond {cond['id']}] {cond['strategy']:13s} n={cond['n_agents']:5d} "
                      f"m={cond['nb_mushrooms']:5d} p={cond['poison_multiplier']:+.1f} seed={seed}  "
                      f"(already done, skipping)  tau={row['tau_mean']:8.2f}  ratio={row['ratio']:+.3f}")
                continue

            # each run is its own subprocess, not an in-process call: running
            # many distinct (strategy, n_agents, nb_mushrooms) shapes in one
            # long-lived process accumulates compiled-program memory and can
            # OOM the machine -- reproduced this directly on a 45-run sweep.
            # The persistent compilation cache in calibrate.py (see its
            # header) keeps this fast despite the process-per-run overhead:
            # only the first seed of each shape actually pays compile time.
            subprocess.run(
                [sys.executable, CALIBRATE_PY,
                 "--strategy", cond["strategy"],
                 "--n_agents", str(cond["n_agents"]),
                 "--nb_mushrooms", str(cond["nb_mushrooms"]),
                 "--poison_multiplier", str(cond["poison_multiplier"]),
                 "--seed", str(seed),
                 "--grid_size", str(COMMON["grid_size"]),
                 "--nutrition", str(COMMON["nutrition"]),
                 "--energy_decay", str(COMMON["energy_decay"]),
                 "--energy_start", str(COMMON["energy_start"]),
                 "--perc_radius", str(COMMON["perc_radius"]),
                 "--steps", str(COMMON["steps"]),
                 "--epsilon", str(COMMON["epsilon"]),
                 "--k", str(COMMON["k"]),
                 "--out", OUT_DIR],
                check=True,
            )
            row = analyze_run(path)
            row["condition_id"] = cond["id"]
            rows.append(row)
            print(f"[cond {cond['id']}] {cond['strategy']:13s} n={cond['n_agents']:5d} "
                  f"m={cond['nb_mushrooms']:5d} p={cond['poison_multiplier']:+.1f} seed={seed}  "
                  f"tau={row['tau_mean']:8.2f}  tau_e={row['tau_e_mean']:8.2f}  "
                  f"f_poison={row['f_poison']:.4f}  g_measured={row['g_measured']:+.4f}  "
                  f"g_predicted={row['g_predicted']:+.4f}  ratio={row['ratio']:+.3f}")

    fieldnames = ["condition_id"] + [k for k in rows[0].keys() if k != "condition_id"]
    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {SUMMARY_PATH}")

    run_checks(rows)


def run_checks(rows):
    print("\n=== checks ===")
    failures = []

    def check(label, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(label)

    lone_approach = [r for r in rows if r["strategy"] == "approach_all" and r["n_agents"] == 1]
    for r in lone_approach:
        ok = 0.45 <= r["f_poison"] <= 0.55
        check(f"approach_all f_poison in [0.45,0.55] (cond {r['condition_id']} seed {r['seed']})",
              ok, f"f_poison={r['f_poison']:.4f}")

    lone_disc = [r for r in rows if r["strategy"] == "discriminate" and r["n_agents"] == 1]
    for r in lone_disc:
        ok = r["f_poison"] < 0.02
        check(f"discriminate f_poison < 0.02 (cond {r['condition_id']} seed {r['seed']})",
              ok, f"f_poison={r['f_poison']:.4f}")

    cond2 = [r for r in rows if r["condition_id"] == 2]
    for r in cond2:
        energy_decay = COMMON["energy_decay"]
        diff = r["g_measured"] - (-energy_decay)
        check(f"approach_all P=-1.0: g_measured ~= -energy_decay (cond 2 seed {r['seed']})",
              abs(diff) < 0.02, f"g_measured={r['g_measured']:+.4f}, diff={diff:+.4f}")

    cond3 = [r for r in rows if r["condition_id"] == 3]
    for r in cond3:
        check(f"approach_all P=-0.5: g_measured > 0 (cond 3 seed {r['seed']})",
              r["g_measured"] > 0, f"g_measured={r['g_measured']:+.4f}")

    lone_runs = [r for r in rows if r["n_agents"] == 1 and r["strategy"] in ("approach_all", "discriminate")]
    for r in lone_runs:
        # ratio is meaningless (holds a difference instead) when g_predicted ~ 0 -- see calibrate_analysis.py
        near_zero_gpred = abs(r["g_predicted"]) < 0.01
        if near_zero_gpred:
            check(f"ratio check skipped, g_predicted~0 (cond {r['condition_id']} seed {r['seed']})",
                  True, f"g_measured-g_predicted={r['ratio']:+.4f} reported instead")
        else:
            ok = 0.8 <= r["ratio"] <= 1.2
            check(f"ratio in [0.8,1.2] (cond {r['condition_id']} seed {r['seed']}, {r['strategy']})",
                  ok, f"ratio={r['ratio']:+.4f}")

    wander_rows = [r for r in rows if r["strategy"] == "wander"]
    for r in wander_rows:
        check(f"wander encounter_rate < 0.02 (seed {r['seed']})",
              r["encounter_rate"] < 0.02, f"encounter_rate={r['encounter_rate']:.5f}")

    print(f"\n{len(failures)} check(s) failed out of "
          f"{len(lone_approach)+len(lone_disc)+len(cond2)+len(cond3)+len(lone_runs)+len(wander_rows)}")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
