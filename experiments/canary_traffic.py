"""Try this 1: how much traffic should the canary get? (make try-canary-traffic)

simulate_canary.py sends a share of the requests to the candidate and decides once the
candidate has served at least 10 of them. A bigger share decides sooner, and exposes more
requests to a bad candidate. This runs the script's own CanaryDeploymentSimulator with shares
of 5, 10, 20 and 50% and 10 routing seeds each, for both candidates, up to 400 requests.

To keep it fast, each model's action and latency are measured once per test frame, and the
simulator reads them from that table instead of running the model again. Routing, metrics and
the rollback rules are the script's. No server, no Docker.
"""

import logging
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
logging.disable(logging.WARNING)
import simulate_canary as sc  # noqa: E402

SHARES = (5, 10, 20, 50)
SEEDS = range(10)


def measured(manager, requests):
    """{frame: (action, latency)} for every test request, measured once."""
    model, table = manager.get_model(), {}
    for rgb, pkl, _ in requests:
        start = time.perf_counter()
        action = model.predict(rgb, pkl)
        table[rgb] = (action, time.perf_counter() - start)
    return table


def main():
    tables = {}
    reasons = Counter()
    for candidate in ("manifest_candidate.yaml", "manifest_candidate_no_camera.yaml"):
        print(f"Candidate {candidate}: up to 400 requests, 10 seeds per share\n")
        print(f"{'canary share':>12} {'decisions':<28} {'requests to decide':>19} {'served by candidate':>20}")
        for share in SHARES:
            decisions, requests_used, exposed = Counter(), [], []
            for seed in SEEDS:
                sim = sc.CanaryDeploymentSimulator("manifest.yaml", candidate, sc.CanaryConfig(
                    canary_traffic_percentage=share, max_samples=400, seed=seed))
                if not tables:
                    reqs = sim.find_test_requests()
                    tables[sim.baseline_model.get_version()] = measured(sim.baseline_model, reqs)
                for m in (sim.candidate_model,):
                    if m.get_version() not in tables:
                        tables[m.get_version()] = measured(m, sim.find_test_requests())
                sim.run_inference = lambda manager, rgb, pkl: tables[manager.get_version()][rgb]
                r = sim.run_simulation()
                decisions[r["decision"]] += 1
                if r["decision"] == "rollback":
                    reasons[r["rollback_reason"].split(":")[0]] += 1
                if r["decision"] != "continue":
                    requests_used.append(r["samples_processed"])
                exposed.append(r["final_metrics"]["candidate"]["samples"])
            text = ", ".join(f"{n} {d}" for d, n in sorted(decisions.items()))
            median = f"{np.median(requests_used):.0f}" if requests_used else "-"
            print(f"{share:>11}% {text:<28} {median:>19} {np.median(exposed):>20.0f}")
        print(f"rollback reasons: {dict(reasons)}\n")
        reasons.clear()
    print("requests to decide: median over the seeds that decided. served by candidate: median requests\n"
          "the candidate answered before the decision (or in all 400, when it was promoted).")


if __name__ == "__main__":
    main()
