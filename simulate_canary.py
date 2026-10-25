#!/usr/bin/env python3
"""
Canary deployment simulation script.

This script simulates a canary deployment on held-out RLBench demonstrations:
1. Loads the baseline (manifest.yaml, v1.0.0) and a candidate (--candidate, v1.1.0 by default)
2. Replays requests (one camera frame plus telemetry each) and routes a share to the candidate
3. Compares the two on the candidate's requests: latency, errors, error against what the expert
   actually did next, and how far the candidate's actions drift from the baseline's (the baseline
   runs in shadow on those requests; its answer isn't served)
4. Rolls back as soon as the candidate is clearly worse; promotes it if it holds up

    python simulate_canary.py --max-samples 10 --canary-traffic 20.0
    python simulate_canary.py --max-samples 400 --canary-traffic 20
    python simulate_canary.py --candidate manifest_candidate_no_camera.yaml --max-samples 400 --canary-traffic 20

Both models are trained first if their weights are missing (python train.py).
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import rlbench_data
from model import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    canary_traffic_percentage: float = 10.0  # Percentage of traffic to route to candidate
    latency_threshold_multiplier: float = 1.5  # Max allowed p95 latency increase
    action_divergence_threshold: float = 0.2  # Max allowed action divergence from the baseline
    quality_tolerance: float = 0.10  # Max allowed increase of the error against expert actions
    min_samples: int = 10  # Minimum candidate samples before making a decision
    max_samples: int = 100  # Maximum requests to replay
    seed: int = 0


@dataclass
class ModelMetrics:
    """Metrics for a model version."""
    latencies: List[float] = field(default_factory=list)
    errors_vs_expert: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def p95_latency(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else float('inf')

    @property
    def mae(self) -> float:
        return float(np.mean(self.errors_vs_expert)) if self.errors_vs_expert else float('inf')

    @property
    def success_rate(self) -> float:
        total = len(self.latencies) + self.errors
        return len(self.latencies) / total if total > 0 else 0.0

    def summary(self) -> Dict:
        return {"samples": len(self.latencies), "errors": self.errors, "success_rate": self.success_rate,
                "p50_latency_s": float(np.median(self.latencies)) if self.latencies else None,
                "p95_latency_s": self.p95_latency if self.latencies else None,
                "mae_vs_expert_rad": self.mae if self.errors_vs_expert else None}


class CanaryDeploymentSimulator:
    """Simulates canary deployment with real RLBench data."""

    def __init__(self, baseline_manifest: str, candidate_manifest: str, config: CanaryConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Load models
        self.baseline_model = ModelManager(baseline_manifest)
        self.candidate_model = ModelManager(candidate_manifest)

        # Metrics tracking: baseline on its own traffic, candidate on its traffic,
        # and the baseline in shadow on the candidate's traffic (for comparison only)
        self.baseline_metrics = ModelMetrics()
        self.candidate_metrics = ModelMetrics()
        self.shadow_metrics = ModelMetrics()
        self.divergences: List[float] = []

        logger.info("Initialized canary simulation:")
        logger.info(f"  Baseline: {self.baseline_model.get_version()}")
        logger.info(f"  Candidate: {self.candidate_model.get_version()}")
        logger.info(f"  Canary traffic: {config.canary_traffic_percentage}%")

    def find_test_requests(self) -> List[Tuple[str, str, np.ndarray]]:
        """Requests from demonstrations neither model trained on: (frame, telemetry, expert action)."""
        training = self.baseline_model.get_config().get("training", {})
        dirs = rlbench_data.require_data(training.get("task", rlbench_data.DEFAULT_TASK))
        requests = []
        for episode_dir in rlbench_data.parse_episodes(training.get("test_episodes", "20-24"), dirs):
            pkl_path = os.path.join(episode_dir, "low_dim_obs.pkl")
            actions = rlbench_data.expert_actions(rlbench_data.load_demo(pkl_path)["joint_positions"])
            requests += [(rlbench_data.frame_path(episode_dir, step), pkl_path, action)
                         for step, action in enumerate(actions)]
        logger.info(f"Found {len(requests)} test requests")
        return requests

    def run_inference(self, model_manager: ModelManager, rgb_path: str, pkl_path: str) -> Tuple[np.ndarray, float]:
        """Run inference and measure latency (reading the frame and telemetry included)."""
        start_time = time.perf_counter()
        action = model_manager.get_model().predict(rgb_path, pkl_path)
        return action, time.perf_counter() - start_time

    def should_rollback(self) -> Tuple[bool, str]:
        """Determine if we should rollback based on metrics."""
        candidate, shadow = self.candidate_metrics, self.shadow_metrics
        if len(candidate.latencies) + candidate.errors < self.config.min_samples:
            return False, "Insufficient samples"

        # Check success rates
        if candidate.success_rate < self.baseline_metrics.success_rate * 0.95:  # 5% tolerance
            return True, f"Success rate degradation: {candidate.success_rate:.3f} vs {self.baseline_metrics.success_rate:.3f}"

        # Check latency degradation (same requests, baseline in shadow)
        if candidate.p95_latency > shadow.p95_latency * self.config.latency_threshold_multiplier:
            return True, f"Latency degradation: p95 {candidate.p95_latency * 1000:.1f} ms vs {shadow.p95_latency * 1000:.1f} ms"

        # Check quality against what the expert did
        if candidate.mae > shadow.mae * (1 + self.config.quality_tolerance):
            return True, (f"Worse actions: error vs expert {candidate.mae:.5f} rad vs {shadow.mae:.5f} rad "
                          f"(+{(candidate.mae / shadow.mae - 1) * 100:.0f}%)")

        # Check action divergence from the baseline
        divergence = self.action_divergence()
        if divergence > self.config.action_divergence_threshold:
            return True, f"Action divergence too high: {divergence:.3f}"

        return False, "Metrics within acceptable range"

    def action_divergence(self) -> float:
        """Mean squared difference between candidate and baseline actions on the same requests,
        relative to the mean squared baseline action."""
        if not self.divergences:
            return 0.0
        squared_difference, squared_baseline = np.array(self.divergences).sum(axis=0)
        return float(squared_difference / (squared_baseline + 1e-12))

    def run_simulation(self) -> Dict:
        """Run the canary deployment simulation."""
        logger.info("Starting canary deployment simulation...")

        requests = self.find_test_requests()
        if not requests:
            raise ValueError("No test requests found")
        order = self.rng.permutation(len(requests))[:self.config.max_samples]

        results = {
            "baseline_version": self.baseline_model.get_version(),
            "candidate_version": self.candidate_model.get_version(),
            "config": self.config.__dict__,
            "decision": "continue",
            "rollback_triggered": False,
            "rollback_reason": "",
            "samples_processed": 0,
            "final_metrics": {}
        }

        for i, index in enumerate(order):
            rgb_path, pkl_path, expert = requests[index]
            # Determine which model serves this request
            use_candidate = self.rng.random() < (self.config.canary_traffic_percentage / 100.0)
            metrics = self.candidate_metrics if use_candidate else self.baseline_metrics
            try:
                if use_candidate:
                    action, latency = self.run_inference(self.candidate_model, rgb_path, pkl_path)
                    shadow_action, shadow_latency = self.run_inference(self.baseline_model, rgb_path, pkl_path)
                    self.shadow_metrics.latencies.append(shadow_latency)
                    self.shadow_metrics.errors_vs_expert.append(float(np.abs(shadow_action - expert).mean()))
                    self.divergences.append((float(((action - shadow_action) ** 2).sum()),
                                             float((shadow_action ** 2).sum())))
                else:
                    action, latency = self.run_inference(self.baseline_model, rgb_path, pkl_path)
                metrics.latencies.append(latency)
                metrics.errors_vs_expert.append(float(np.abs(action - expert).mean()))
                logger.debug(f"Sample {i+1}: {'candidate' if use_candidate else 'baseline'}, {latency * 1000:.1f} ms")
            except Exception as e:
                metrics.errors += 1
                logger.error(f"Sample {i+1} failed: {e}")
            results["samples_processed"] = i + 1

            if use_candidate:
                should_rollback, reason = self.should_rollback()
                if should_rollback:
                    logger.warning(f"ROLLBACK TRIGGERED after {i + 1} requests: {reason}")
                    results.update(decision="rollback", rollback_triggered=True, rollback_reason=reason)
                    break

        if not results["rollback_triggered"]:
            enough = len(self.candidate_metrics.latencies) + self.candidate_metrics.errors >= self.config.min_samples
            results["decision"] = "promote" if enough else "continue"

        # Final metrics
        results["final_metrics"] = {
            "baseline": self.baseline_metrics.summary(),
            "candidate": self.candidate_metrics.summary(),
            "baseline_on_candidate_requests": self.shadow_metrics.summary(),
            "action_divergence": self.action_divergence(),
        }
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", default="manifest_candidate.yaml",
                        help="candidate manifest (default: manifest_candidate.yaml)")
    parser.add_argument("--canary-traffic", type=float, default=10.0,
                        help="Percentage of traffic to route to candidate (default: 10.0)")
    parser.add_argument("--latency-threshold", type=float, default=1.5,
                        help="Latency threshold multiplier (default: 1.5)")
    parser.add_argument("--divergence-threshold", type=float, default=0.2,
                        help="Action divergence threshold (default: 0.2)")
    parser.add_argument("--quality-tolerance", type=float, default=0.10,
                        help="Allowed increase of the error against expert actions (default: 0.10)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Candidate samples needed before deciding (default: 10)")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Maximum samples to test (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="Traffic routing seed (default: 0)")
    args = parser.parse_args()

    # Create configuration
    config = CanaryConfig(
        canary_traffic_percentage=args.canary_traffic,
        latency_threshold_multiplier=args.latency_threshold,
        action_divergence_threshold=args.divergence_threshold,
        quality_tolerance=args.quality_tolerance,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Run simulation
    simulator = CanaryDeploymentSimulator("manifest.yaml", args.candidate, config)
    results = simulator.run_simulation()

    # Print results
    print("\n" + "=" * 60)
    print("CANARY DEPLOYMENT SIMULATION RESULTS")
    print("=" * 60)

    print(f"Baseline Version: {results['baseline_version']}")
    print(f"Candidate Version: {results['candidate_version']} ({args.candidate})")
    print(f"Samples Processed: {results['samples_processed']}")

    final = results['final_metrics']
    if results['rollback_triggered']:
        print(f"\nROLLBACK TRIGGERED: {results['rollback_reason']}")
    elif results['decision'] == 'promote':
        print("\nCanary deployment SUCCESSFUL - no rollback needed; the candidate can be promoted")
    else:
        print(f"\nNot enough candidate traffic to decide ({final['candidate']['samples']} of "
              f"{config.min_samples} samples): the baseline keeps serving; run with more samples")

    def row(name, m):
        if not m['samples']:
            return f"  {name:32s} 0 samples"
        return (f"  {name:32s} {m['samples']:4d} samples, p95 latency {m['p95_latency_s'] * 1000:6.1f} ms, "
                f"error vs expert {m['mae_vs_expert_rad']:.5f} rad, success {m['success_rate']:.3f}")

    print("\nFinal Metrics:")
    print(row("Baseline", final['baseline']))
    print(row("Candidate", final['candidate']))
    print(row("Baseline on candidate requests", final['baseline_on_candidate_requests']))
    print(f"  Action Divergence: {final['action_divergence']:.3f}")

    # Save detailed results
    with open("canary_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDetailed results saved to canary_results.json")


if __name__ == "__main__":
    main()
