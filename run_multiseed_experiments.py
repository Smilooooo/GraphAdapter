"""
Multi-seed GraphAdapter Experiments

Runs GraphAdapter with multiple backbones across different seeds and aggregates results.

Usage:
    python run_multiseed_experiments.py \
        --root "E:\BachelorThesis\Data\data(3)\data" \
        --dataset-config-file configs/datasets/lunghist700.yaml \
        --num-shots 4 \
        --seeds 11111 22222 33333 \
        --backbones biomedclip plip conch \
        --output-base output/lunghist700_multiseed
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Backbone to config file mapping
BACKBONE_CONFIGS = {
    "biomedclip": "configs/trainers/GraphCLIP_v2/biomedclip.yaml",
    "plip": "configs/trainers/GraphCLIP_v2/plip.yaml",
    "conch": "configs/trainers/GraphCLIP_v2/conch.yaml",
    "rn50": "configs/trainers/GraphCLIP_v2/rn50.yaml",
}


def parse_log_file(log_path):
    """Extract test accuracy from log file."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Look for the final test result
        for line in reversed(lines):
            if '* accuracy:' in line.lower():
                # Extract percentage (e.g., "* accuracy: 68.1%")
                parts = line.split(':')
                if len(parts) >= 2:
                    acc_str = parts[1].strip().replace('%', '')
                    return float(acc_str)
    except Exception as e:
        print(f"  Warning: Could not parse log file {log_path}: {e}")
    return None


def run_single_experiment(args, backbone, seed, output_dir):
    """Run a single training experiment."""
    print(f"\n{'='*70}")
    print(f"Backbone: {backbone.upper()} | Seed: {seed}")
    print(f"{'='*70}")
    
    config_file = BACKBONE_CONFIGS[backbone]
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        "train.py",
        "--root", args.root,
        "--dataset-config-file", args.dataset_config_file,
        "--config-file", config_file,
        "--trainer", "GraphCLIP_v2",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "DATASET.NUM_SHOTS", str(args.num_shots),
    ]
    
    # Add HF_TOKEN for CONCH
    env = os.environ.copy()
    if backbone == "conch" and args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}\n")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        print(f"\n✓ Completed: {backbone} (seed {seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {backbone} (seed {seed})")
        print(f"Error: {e}")
        return False


def aggregate_results(args, results):
    """Compute mean and std across seeds for each backbone."""
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (Mean ± Std)")
    print(f"{'='*70}\n")
    
    aggregated = {}
    
    for backbone in args.backbones:
        accuracies = [r['accuracy'] for r in results if r['backbone'] == backbone and r['accuracy'] is not None]
        
        if len(accuracies) > 0:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            aggregated[backbone] = {
                'mean': mean_acc,
                'std': std_acc,
                'accuracies': accuracies,
                'n_seeds': len(accuracies)
            }
            print(f"{backbone:15s}: {mean_acc:.2f}% ± {std_acc:.2f}%  (n={len(accuracies)})")
        else:
            print(f"{backbone:15s}: No valid results")
            aggregated[backbone] = None
    
    return aggregated


def save_summary(args, results, aggregated, output_file):
    """Save results summary to file."""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GraphAdapter Multi-Seed Experiment Results\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_config_file}\n")
        f.write(f"Num shots: {args.num_shots}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Backbones: {args.backbones}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Individual Results\n")
        f.write("="*70 + "\n\n")
        
        for r in results:
            status = f"{r['accuracy']:.2f}%" if r['accuracy'] else "FAILED"
            f.write(f"{r['backbone']:15s} | Seed {r['seed']:6d} | {status:10s} | {r['output_dir']}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Aggregated Results (Mean ± Std)\n")
        f.write("="*70 + "\n\n")
        
        for backbone in args.backbones:
            if aggregated[backbone]:
                agg = aggregated[backbone]
                f.write(f"{backbone:15s}: {agg['mean']:.2f}% ± {agg['std']:.2f}%  (n={agg['n_seeds']})\n")
                f.write(f"                  Individual: {', '.join([f'{a:.2f}%' for a in agg['accuracies']])}\n")
            else:
                f.write(f"{backbone:15s}: No valid results\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed GraphAdapter experiments")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--dataset-config-file", type=str, required=True,
                        help="Path to dataset config")
    parser.add_argument("--num-shots", type=int, default=4,
                        help="Number of shots per class")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11111, 22222, 33333],
                        help="Random seeds to use")
    parser.add_argument("--backbones", type=str, nargs="+",
                        default=["biomedclip", "plip", "conch"],
                        help="Backbones to test")
    parser.add_argument("--output-base", type=str, default="output/multiseed",
                        help="Base output directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (required for CONCH)")
    
    args = parser.parse_args()
    
    # Validate backbones
    for backbone in args.backbones:
        if backbone not in BACKBONE_CONFIGS:
            print(f"Error: Unknown backbone '{backbone}'")
            print(f"Available: {list(BACKBONE_CONFIGS.keys())}")
            sys.exit(1)
    
    # Create output base directory
    os.makedirs(args.output_base, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print("# GraphAdapter Multi-Seed Experiments")
    print(f"# Seeds: {args.seeds}")
    print(f"# Backbones: {args.backbones}")
    print(f"# Shots: {args.num_shots}")
    print(f"{'#'*70}")
    
    # Run all experiments
    results = []
    total = len(args.backbones) * len(args.seeds)
    current = 0
    
    for backbone in args.backbones:
        for seed in args.seeds:
            current += 1
            print(f"\n[{current}/{total}] Starting: {backbone} with seed {seed}")
            
            output_dir = os.path.join(
                args.output_base,
                f"{backbone}_seed{seed}"
            )
            
            success = run_single_experiment(args, backbone, seed, output_dir)
            
            # Parse results from log file
            log_file = os.path.join(output_dir, "log.txt")
            accuracy = parse_log_file(log_file) if success else None
            
            results.append({
                'backbone': backbone,
                'seed': seed,
                'accuracy': accuracy,
                'success': success,
                'output_dir': output_dir
            })
    
    # Aggregate results
    print(f"\n{'='*70}")
    print(f"Completed {current}/{total} experiments")
    print(f"{'='*70}")
    
    aggregated = aggregate_results(args, results)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_base, f"summary_{timestamp}.txt")
    save_summary(args, results, aggregated, summary_file)
    
    # Also save as JSON for easy parsing
    json_file = os.path.join(args.output_base, f"results_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'aggregated': {k: v for k, v in aggregated.items() if v}
        }, f, indent=2)
    print(f"JSON results saved to: {json_file}")


if __name__ == "__main__":
    main()
