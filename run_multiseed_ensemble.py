"""
Multi-seed Ensemble Experiments for GraphAdapter

Runs ensemble training/evaluation across multiple seeds and aggregates results.

Usage:
    $env:HF_TOKEN = "your_token"; python run_multiseed_ensemble.py \
        --root "E:\BachelorThesis\Data\data(3)\data" \
        --dataset-config-file configs/datasets/lunghist700.yaml \
        --num-shots 4 \
        --seeds 11111 22222 33333 \
        --backbones biomedclip plip conch \
        --output-base output/ensemble_multiseed
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def parse_ensemble_results(results_file):
    """Parse ensemble_results.txt file."""
    try:
        with open(results_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        results = {
            'individual': {},
            'ensemble': None
        }
        
        # Parse individual accuracies
        in_individual_section = False
        for line in lines:
            if 'Individual Accuracies:' in line:
                in_individual_section = True
                continue
            elif 'Ensemble Accuracy' in line:
                in_individual_section = False
                # Extract ensemble accuracy
                parts = line.split(':')
                if len(parts) >= 2:
                    acc_str = parts[-1].strip().replace('%', '')
                    results['ensemble'] = float(acc_str)
            elif in_individual_section and ':' in line:
                # Parse individual line like "  biomedclip: 62.32%"
                parts = line.strip().split(':')
                if len(parts) == 2:
                    backbone = parts[0].strip()
                    acc_str = parts[1].strip().replace('%', '')
                    try:
                        results['individual'][backbone] = float(acc_str)
                    except ValueError:
                        pass
        
        return results
    except Exception as e:
        print(f"  Warning: Could not parse results file {results_file}: {e}")
        return None


def run_ensemble_experiment(args, seed, output_dir):
    """Run ensemble training for a single seed."""
    print(f"\n{'='*70}")
    print(f"Ensemble Experiment | Seed: {seed}")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        sys.executable,
        "ensemble_train_eval.py",
        "--root", args.root,
        "--dataset-config-file", args.dataset_config_file,
        "--num-shots", str(args.num_shots),
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--backbones"
    ] + args.backbones
    
    # Add HF_TOKEN for CONCH
    env = os.environ.copy()
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}\n")
    
    # Run ensemble training
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        print(f"\n✓ Completed ensemble with seed {seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed ensemble with seed {seed}")
        print(f"Error: {e}")
        return False


def aggregate_results(args, all_results):
    """Compute mean and std across seeds."""
    print(f"\n{'='*70}")
    print("AGGREGATED ENSEMBLE RESULTS (Mean ± Std)")
    print(f"{'='*70}\n")
    
    # Aggregate individual model results
    print("Individual Models:")
    individual_aggregated = {}
    for backbone in args.backbones:
        accuracies = []
        for r in all_results:
            if r['results'] and backbone in r['results']['individual']:
                accuracies.append(r['results']['individual'][backbone])
        
        if len(accuracies) > 0:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            individual_aggregated[backbone] = {
                'mean': mean_acc,
                'std': std_acc,
                'accuracies': accuracies,
                'n_seeds': len(accuracies)
            }
            print(f"  {backbone:15s}: {mean_acc:.2f}% ± {std_acc:.2f}%  (n={len(accuracies)})")
        else:
            print(f"  {backbone:15s}: No valid results")
            individual_aggregated[backbone] = None
    
    # Aggregate ensemble results
    print("\nEnsemble (Majority Voting):")
    ensemble_accuracies = [r['results']['ensemble'] for r in all_results 
                          if r['results'] and r['results']['ensemble'] is not None]
    
    if len(ensemble_accuracies) > 0:
        mean_acc = np.mean(ensemble_accuracies)
        std_acc = np.std(ensemble_accuracies)
        ensemble_aggregated = {
            'mean': mean_acc,
            'std': std_acc,
            'accuracies': ensemble_accuracies,
            'n_seeds': len(ensemble_accuracies)
        }
        print(f"  Ensemble        : {mean_acc:.2f}% ± {std_acc:.2f}%  (n={len(ensemble_accuracies)})")
    else:
        print(f"  Ensemble        : No valid results")
        ensemble_aggregated = None
    
    return individual_aggregated, ensemble_aggregated


def save_summary(args, all_results, individual_agg, ensemble_agg, output_file):
    """Save results summary to file."""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GraphAdapter Multi-Seed Ensemble Results\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset_config_file}\n")
        f.write(f"Num shots: {args.num_shots}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Backbones: {args.backbones}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("Individual Results per Seed\n")
        f.write("="*70 + "\n\n")
        
        for r in all_results:
            f.write(f"Seed {r['seed']}:\n")
            if r['results']:
                f.write(f"  Individual models:\n")
                for backbone in args.backbones:
                    if backbone in r['results']['individual']:
                        acc = r['results']['individual'][backbone]
                        f.write(f"    {backbone:15s}: {acc:.2f}%\n")
                if r['results']['ensemble']:
                    f.write(f"  Ensemble          : {r['results']['ensemble']:.2f}%\n")
            else:
                f.write(f"  FAILED\n")
            f.write(f"  Output: {r['output_dir']}\n\n")
        
        f.write("="*70 + "\n")
        f.write("Aggregated Results (Mean ± Std)\n")
        f.write("="*70 + "\n\n")
        
        f.write("Individual Models:\n")
        for backbone in args.backbones:
            if individual_agg[backbone]:
                agg = individual_agg[backbone]
                f.write(f"  {backbone:15s}: {agg['mean']:.2f}% ± {agg['std']:.2f}%  (n={agg['n_seeds']})\n")
                f.write(f"                    Individual: {', '.join([f'{a:.2f}%' for a in agg['accuracies']])}\n")
            else:
                f.write(f"  {backbone:15s}: No valid results\n")
        
        f.write("\nEnsemble (Majority Voting):\n")
        if ensemble_agg:
            f.write(f"  Ensemble        : {ensemble_agg['mean']:.2f}% ± {ensemble_agg['std']:.2f}%  (n={ensemble_agg['n_seeds']})\n")
            f.write(f"                    Individual: {', '.join([f'{a:.2f}%' for a in ensemble_agg['accuracies']])}\n")
        else:
            f.write(f"  Ensemble        : No valid results\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed ensemble experiments")
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
                        help="Backbones to include in ensemble")
    parser.add_argument("--output-base", type=str, default="output/ensemble_multiseed",
                        help="Base output directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (required for CONCH)")
    
    args = parser.parse_args()
    
    # Create output base directory
    os.makedirs(args.output_base, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print("# GraphAdapter Multi-Seed Ensemble Experiments")
    print(f"# Seeds: {args.seeds}")
    print(f"# Backbones: {args.backbones}")
    print(f"# Shots: {args.num_shots}")
    print(f"{'#'*70}")
    
    # Run ensemble for each seed
    all_results = []
    
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n[{i}/{len(args.seeds)}] Starting ensemble with seed {seed}")
        
        output_dir = os.path.join(args.output_base, f"seed_{seed}")
        
        success = run_ensemble_experiment(args, seed, output_dir)
        
        # Parse results
        results_file = os.path.join(output_dir, "ensemble_results.txt")
        results = parse_ensemble_results(results_file) if success else None
        
        all_results.append({
            'seed': seed,
            'success': success,
            'results': results,
            'output_dir': output_dir
        })
    
    # Aggregate results
    print(f"\n{'='*70}")
    print(f"Completed {len(args.seeds)} ensemble experiments")
    print(f"{'='*70}")
    
    individual_agg, ensemble_agg = aggregate_results(args, all_results)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_base, f"ensemble_summary_{timestamp}.txt")
    save_summary(args, all_results, individual_agg, ensemble_agg, summary_file)
    
    # Save as JSON
    json_file = os.path.join(args.output_base, f"ensemble_results_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'all_results': all_results,
            'aggregated': {
                'individual': {k: v for k, v in individual_agg.items() if v},
                'ensemble': ensemble_agg
            }
        }, f, indent=2)
    print(f"JSON results saved to: {json_file}")


if __name__ == "__main__":
    main()
