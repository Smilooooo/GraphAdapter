"""
Systematic experiment runner for reproducing paper results.
Runs multiple seeds and shot configurations for DTD, Caltech101, and EuroSAT.

This script will run:
- 3 datasets: DTD, Caltech101, EuroSAT
- 3 shot settings: 1-shot, 2-shot, 4-shot
- 3 seeds: 1, 2, 3
Total: 27 experiments
"""

import os
import subprocess
import time
from datetime import datetime

# Configuration
DATASETS = {
    "dtd": "DescribableTextures (47 classes)",
    "caltech101": "Caltech101 (101 classes)", 
    "eurosat": "EuroSAT (10 classes)"
}

SHOTS = [1, 2, 4]
SEEDS = [1, 2, 3]

ROOT_DIR = "C:\\datasets"
TRAINER = "GraphCLIP_v1"
CONFIG_FILE = "configs/trainers/GraphCLIP_v1/rn50_ep20_b256_lr_0_001_adamw.yaml"

def run_experiment(dataset, shots, seed):
    """Run a single experiment."""
    
    dataset_config = f"configs/datasets/{dataset}.yaml"
    # Structure: output/{dataset}/{TRAINER}/{shots}shot/seed{seed}
    # This allows parse_test_res.py to work: python parse_test_res.py output/{dataset}/{TRAINER}/{shots}shot
    output_dir = f"output\\{dataset}\\{TRAINER}\\{shots}shot\\seed{seed}"
    
    # Check if already completed
    log_file = os.path.join(output_dir, "log.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            if "Finish training" in content and "=> result" in content:
                print(f"  ✓ Already completed: {dataset} {shots}-shot seed{seed}")
                return True
    
    print(f"\n{'='*70}")
    print(f"Running: {dataset.upper()} | {shots}-shot | Seed {seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        "python", "train.py",
        "--root", ROOT_DIR,
        "--seed", str(seed),
        "--trainer", TRAINER,
        "--dataset-config-file", dataset_config,
        "--config-file", CONFIG_FILE,
        "--output-dir", output_dir,
        "DATASET.NUM_SHOTS", str(shots)
    ]
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Completed in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"✗ Failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all experiments."""
    
    print("="*70)
    print("SYSTEMATIC EXPERIMENT RUNNER")
    print("="*70)
    print(f"\nDatasets: {', '.join(DATASETS.keys())}")
    print(f"Shots: {SHOTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {len(DATASETS) * len(SHOTS) * len(SEEDS)}")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT PLAN:")
    print("="*70)
    exp_num = 1
    for dataset in DATASETS.keys():
        for shots in SHOTS:
            for seed in SEEDS:
                print(f"{exp_num:2d}. {dataset:12s} | {shots}-shot | seed{seed}")
                exp_num += 1
    
    response = input("\nProceed with all experiments? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    total_start = time.time()
    results = []
    
    for dataset in DATASETS.keys():
        for shots in SHOTS:
            for seed in SEEDS:
                success = run_experiment(dataset, shots, seed)
                results.append({
                    'dataset': dataset,
                    'shots': shots,
                    'seed': seed,
                    'success': success
                })
    
    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {successful}/{len(results)} experiments")
    
    # Show failures
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\nFailed experiments:")
        for r in failures:
            print(f"  ✗ {r['dataset']} {r['shots']}-shot seed{r['seed']}")
    
    print(f"\nFinish time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)
    print("To parse results, run these commands:")
    print("="*70)
    for dataset in DATASETS.keys():
        for shots in SHOTS:
            print(f"python parse_test_res.py output/{dataset}/{TRAINER}/{shots}shot --ci95")

if __name__ == "__main__":
    main()
