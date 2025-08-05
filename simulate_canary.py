#!/usr/bin/env python3
"""
Canary deployment simulation script.

This script simulates canary deployments by:
1. Loading baseline (v1.0.0) and candidate (v1.1.0) model configurations
2. Routing a percentage of test requests to the candidate model
3. Comparing latency and action distribution between models
4. Rolling back to baseline if divergence exceeds thresholds
"""

import yaml
import time
import numpy as np
import os
import glob
import logging
import argparse
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
from model import ModelManager
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    canary_traffic_percentage: float = 10.0  # Percentage of traffic to route to candidate
    latency_threshold_multiplier: float = 1.5  # Max allowed latency increase
    action_divergence_threshold: float = 0.2  # Max allowed action vector divergence
    min_samples: int = 10  # Minimum samples before making rollback decision
    max_samples: int = 100  # Maximum samples to test

@dataclass
class ModelMetrics:
    """Metrics for a model version."""
    latencies: List[float]
    actions: List[np.ndarray]
    errors: int = 0
    
    @property
    def avg_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else float('inf')
    
    @property
    def success_rate(self) -> float:
        total = len(self.latencies) + self.errors
        return len(self.latencies) / total if total > 0 else 0.0

class CanaryDeploymentSimulator:
    """Simulates canary deployment with real RLBench data."""
    
    def __init__(self, baseline_manifest: str, candidate_manifest: str, config: CanaryConfig):
        self.config = config
        self.baseline_manifest = baseline_manifest
        self.candidate_manifest = candidate_manifest
        
        # Load models
        self.baseline_model = ModelManager(baseline_manifest)
        self.candidate_model = ModelManager(candidate_manifest)
        
        # Metrics tracking
        self.baseline_metrics = ModelMetrics([], [])
        self.candidate_metrics = ModelMetrics([], [])
        
        logger.info(f"Initialized canary simulation:")
        logger.info(f"  Baseline: {self.baseline_model.get_version()}")
        logger.info(f"  Candidate: {self.candidate_model.get_version()}")
        logger.info(f"  Canary traffic: {config.canary_traffic_percentage}%")
    
    def find_test_episodes(self, data_dir: str = "rlbench-data") -> List[Tuple[str, str]]:
        """Find test episodes with RGB and pickle data."""
        episodes = []
        
        # Search for episodes in the data directory
        pattern = os.path.join(data_dir, "*", "*", "episodes", "*")
        episode_dirs = glob.glob(pattern)
        
        for episode_dir in episode_dirs:
            pkl_path = os.path.join(episode_dir, "low_dim_obs.pkl")
            rgb_dir = os.path.join(episode_dir, "front_rgb")
            
            if os.path.exists(pkl_path) and os.path.exists(rgb_dir):
                # Check if RGB directory has images
                rgb_files = glob.glob(os.path.join(rgb_dir, "*.png"))
                if rgb_files:
                    episodes.append((rgb_dir, pkl_path))
        
        logger.info(f"Found {len(episodes)} test episodes")
        return episodes
    
    def run_inference(self, model_manager: ModelManager, rgb_path: str, pkl_path: str) -> Tuple[np.ndarray, float]:
        """Run inference and measure latency."""
        start_time = time.time()
        try:
            model = model_manager.get_model()
            
            # Handle directory paths for RGB by finding the latest image
            actual_rgb_path = rgb_path
            if os.path.isdir(rgb_path):
                from model import find_latest_image
                actual_rgb_path = find_latest_image(rgb_path)
            
            action = model.predict(actual_rgb_path, pkl_path)
            latency = time.time() - start_time
            return action, latency
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def calculate_action_divergence(self) -> float:
        """Calculate divergence between baseline and candidate actions."""
        if not self.baseline_metrics.actions or not self.candidate_metrics.actions:
            return 0.0
        
        # Calculate mean squared difference between action vectors
        baseline_actions = np.array(self.baseline_metrics.actions)
        candidate_actions = np.array(self.candidate_metrics.actions)
        
        # Ensure we have the same number of samples for comparison
        min_samples = min(len(baseline_actions), len(candidate_actions))
        if min_samples == 0:
            return 0.0
        
        baseline_mean = np.mean(baseline_actions[:min_samples], axis=0)
        candidate_mean = np.mean(candidate_actions[:min_samples], axis=0)
        
        # Calculate normalized mean squared error
        mse = np.mean((baseline_mean - candidate_mean) ** 2)
        baseline_norm = np.mean(baseline_mean ** 2)
        
        # Normalized divergence
        divergence = mse / (baseline_norm + 1e-8)
        return divergence
    
    def should_rollback(self) -> Tuple[bool, str]:
        """Determine if we should rollback based on metrics."""
        total_samples = len(self.baseline_metrics.latencies) + len(self.candidate_metrics.latencies)
        
        if total_samples < self.config.min_samples:
            return False, "Insufficient samples"
        
        # Check latency degradation
        baseline_latency = self.baseline_metrics.avg_latency
        candidate_latency = self.candidate_metrics.avg_latency
        
        if candidate_latency > baseline_latency * self.config.latency_threshold_multiplier:
            return True, f"Latency degradation: {candidate_latency:.3f}s vs {baseline_latency:.3f}s"
        
        # Check action divergence
        divergence = self.calculate_action_divergence()
        if divergence > self.config.action_divergence_threshold:
            return True, f"Action divergence too high: {divergence:.3f}"
        
        # Check success rates
        baseline_success = self.baseline_metrics.success_rate
        candidate_success = self.candidate_metrics.success_rate
        
        if candidate_success < baseline_success * 0.95:  # 5% tolerance
            return True, f"Success rate degradation: {candidate_success:.3f} vs {baseline_success:.3f}"
        
        return False, "Metrics within acceptable range"
    
    def run_simulation(self) -> Dict:
        """Run the canary deployment simulation."""
        logger.info("Starting canary deployment simulation...")
        
        # Find test episodes
        episodes = self.find_test_episodes()
        if not episodes:
            raise ValueError("No test episodes found")
        
        # Limit to max samples
        episodes = episodes[:self.config.max_samples]
        np.random.shuffle(episodes)  # Randomize order
        
        results = {
            "baseline_version": self.baseline_model.get_version(),
            "candidate_version": self.candidate_model.get_version(),
            "config": self.config.__dict__,
            "rollback_triggered": False,
            "rollback_reason": "",
            "samples_processed": 0,
            "final_metrics": {}
        }
        
        for i, (rgb_path, pkl_path) in enumerate(episodes):
            # Determine which model to use based on canary percentage
            use_candidate = np.random.random() < (self.config.canary_traffic_percentage / 100.0)
            
            try:
                if use_candidate:
                    action, latency = self.run_inference(self.candidate_model, rgb_path, pkl_path)
                    self.candidate_metrics.latencies.append(latency)
                    self.candidate_metrics.actions.append(action)
                    model_used = "candidate"
                else:
                    action, latency = self.run_inference(self.baseline_model, rgb_path, pkl_path)
                    self.baseline_metrics.latencies.append(latency)
                    self.baseline_metrics.actions.append(action)
                    model_used = "baseline"
                
                logger.info(f"Sample {i+1}/{len(episodes)}: {model_used} model, latency: {latency:.3f}s")
                
            except Exception as e:
                if use_candidate:
                    self.candidate_metrics.errors += 1
                else:
                    self.baseline_metrics.errors += 1
                logger.error(f"Sample {i+1} failed: {e}")
            
            # Check for rollback every few samples
            if (i + 1) >= self.config.min_samples and (i + 1) % 5 == 0:
                should_rollback, reason = self.should_rollback()
                if should_rollback:
                    logger.warning(f"ROLLBACK TRIGGERED: {reason}")
                    results["rollback_triggered"] = True
                    results["rollback_reason"] = reason
                    break
            
            results["samples_processed"] = i + 1
        
        # Final metrics
        results["final_metrics"] = {
            "baseline": {
                "avg_latency": self.baseline_metrics.avg_latency,
                "success_rate": self.baseline_metrics.success_rate,
                "samples": len(self.baseline_metrics.latencies),
                "errors": self.baseline_metrics.errors
            },
            "candidate": {
                "avg_latency": self.candidate_metrics.avg_latency,
                "success_rate": self.candidate_metrics.success_rate,
                "samples": len(self.candidate_metrics.latencies),
                "errors": self.candidate_metrics.errors
            },
            "action_divergence": self.calculate_action_divergence()
        }
        
        return results

def create_candidate_manifest():
    """Create a candidate manifest (v1.1.0) for simulation."""
    candidate_config = {
        "version": "1.1.0",
        "vision": "resnet18",
        "telemetry": "joints_v1",
        "policy": "mlp_stub_v1",
        "fusion": "concat",
        "model_settings": {
            "vision_embedding_dim": 512,
            "telemetry_dim": 7,
            "hidden_dim": 512,  # Increased hidden dim for candidate
            "output_dim": 7
        }
    }
    
    with open("manifest_candidate.yaml", "w") as f:
        yaml.dump(candidate_config, f, default_flow_style=False)
    
    return "manifest_candidate.yaml"

def main():
    parser = argparse.ArgumentParser(description="Run canary deployment simulation")
    parser.add_argument("--canary-traffic", type=float, default=10.0,
                       help="Percentage of traffic to route to candidate (default: 10.0)")
    parser.add_argument("--latency-threshold", type=float, default=1.5,
                       help="Latency threshold multiplier (default: 1.5)")
    parser.add_argument("--divergence-threshold", type=float, default=0.2,
                       help="Action divergence threshold (default: 0.2)")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Maximum samples to test (default: 50)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CanaryConfig(
        canary_traffic_percentage=args.canary_traffic,
        latency_threshold_multiplier=args.latency_threshold,
        action_divergence_threshold=args.divergence_threshold,
        max_samples=args.max_samples
    )
    
    # Create candidate manifest
    candidate_manifest = create_candidate_manifest()
    
    try:
        # Run simulation
        simulator = CanaryDeploymentSimulator("manifest.yaml", candidate_manifest, config)
        results = simulator.run_simulation()
        
        # Print results
        print("\n" + "="*60)
        print("CANARY DEPLOYMENT SIMULATION RESULTS")
        print("="*60)
        
        print(f"Baseline Version: {results['baseline_version']}")
        print(f"Candidate Version: {results['candidate_version']}")
        print(f"Samples Processed: {results['samples_processed']}")
        
        if results['rollback_triggered']:
            print(f"\nROLLBACK TRIGGERED: {results['rollback_reason']}")
        else:
            print("\nCanary deployment SUCCESSFUL - no rollback needed")
        
        print("\nFinal Metrics:")
        baseline = results['final_metrics']['baseline']
        candidate = results['final_metrics']['candidate']
        
        print(f"  Baseline:  {baseline['samples']} samples, "
              f"latency: {baseline['avg_latency']:.3f}s, "
              f"success: {baseline['success_rate']:.3f}")
        print(f"  Candidate: {candidate['samples']} samples, "
              f"latency: {candidate['avg_latency']:.3f}s, "
              f"success: {candidate['success_rate']:.3f}")
        print(f"  Action Divergence: {results['final_metrics']['action_divergence']:.3f}")
        
        # Save detailed results
        with open("canary_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to canary_results.json")
        
    finally:
        # Clean up candidate manifest
        if os.path.exists(candidate_manifest):
            os.remove(candidate_manifest)

if __name__ == "__main__":
    main()