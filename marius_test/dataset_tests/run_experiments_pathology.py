import subprocess
import os
from pathlib import Path
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
ROOT_DATA_DIR = "E:/BachelorThesis/Data/data(3)/data"
TRAINER = "GraphCLIP_v1"
TRAINER_CONFIG = "configs/trainers/GraphCLIP_v1/rn50_ep20_b256_lr_0_001_adamw.yaml"
OUTPUT_BASE = "output"

# pathology datasets
DATASETS = [
    "LungHist700",  
    "BRACS",        
    "BACH",         
]

# Few-shot settings
SHOTS = [1, 2, 4]
SEEDS = [1, 2, 3]

# Reduce workers to avoid memory errors 
NUM_WORKERS = 4

def run_training(dataset, shots, seed):
    """Run a single training experiment"""
    # Config file names must match the dataset class names (lowercase)
    config_map = {
        "LungHist700": "lunghist700",
        "BRACS": "bracs",
        "BACH": "iciar2018",  # BACH folder uses iciar2018 dataset class
    }
    dataset_config = f"configs/datasets/{config_map[dataset]}.yaml"
    output_dir = f"{OUTPUT_BASE}/{dataset.lower()}/GraphCLIP_v1/{shots}shot/seed{seed}"
    
    cmd = [
        "python", "train.py",
        "--root", ROOT_DATA_DIR,
        "--trainer", TRAINER,
        "--dataset-config-file", dataset_config,
        "--config-file", TRAINER_CONFIG,
        "--seed", str(seed),
        "--output-dir", output_dir,
        "DATASET.NUM_SHOTS", str(shots),
        "DATALOADER.NUM_WORKERS", str(NUM_WORKERS),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {dataset} | {shots}-shot | seed={seed}")
    print(f"{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nSUCCESS: {dataset} | {shots}-shot | seed={seed}")
        print(f"   Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)\n")
        return True, duration
    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nFAILED: {dataset} | {shots}-shot | seed={seed}")
        print(f"   Error: {e}")
        print(f"   Duration: {duration:.0f} seconds\n")
        return False, duration
    except KeyboardInterrupt:
        print(f"\nINTERRUPTED: {dataset} | {shots}-shot | seed={seed}")
        raise

def main():
    """Run all experiments"""
    total_experiments = len(DATASETS) * len(SHOTS) * len(SEEDS)
    completed = 0
    failed = 0
    skipped = 0
    total_time = 0
    
    results_log = []
    
    print(f"\n{'='*80}")
    print(f"STARTING BATCH EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Datasets: {len(DATASETS)} ({', '.join(DATASETS)})")
    print(f"Shots: {len(SHOTS)} ({', '.join(map(str, SHOTS))})")
    print(f"Seeds: {len(SEEDS)} ({', '.join(map(str, SEEDS))})")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    try:
        # Run all combinations
        for dataset in DATASETS:
            for shots in SHOTS:
                for seed in SEEDS:
                    # Check if already exists (optional - skip completed experiments)
                    output_dir = f"{OUTPUT_BASE}/{dataset.lower()}/GraphCLIP_v1/{shots}shot/seed{seed}"
                    log_file = os.path.join(output_dir, "log.txt")
                    
                    if os.path.exists(log_file):
                        print(f"SKIPPING: {dataset} | {shots}-shot | seed={seed} (already exists)")
                        skipped += 1
                        results_log.append(f"SKIPPED: {dataset} {shots}-shot seed{seed}")
                        continue
                    
                    success, duration = run_training(dataset, shots, seed)
                    total_time += duration
                    
                    if success:
                        completed += 1
                        results_log.append(f"SUCCESS: {dataset} {shots}-shot seed{seed}")
                    else:
                        failed += 1
                        results_log.append(f"FAILED: {dataset} {shots}-shot seed{seed}")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print(f"BATCH EXPERIMENTS INTERRUPTED BY USER")
        print(f"{'='*80}\n")
    
    # Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"BATCH EXPERIMENTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"{'='*80}\n")
    
    # Save detailed summary to file
    summary_file = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, "w") as f:
        f.write(f"GraphAdapter Pathology Experiment Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Datasets: {', '.join(DATASETS)}\n")
        f.write(f"  Shots: {', '.join(map(str, SHOTS))}\n")
        f.write(f"  Seeds: {', '.join(map(str, SEEDS))}\n")
        f.write(f"  Trainer: {TRAINER}\n")
        f.write(f"  Config: {TRAINER_CONFIG}\n")
        f.write(f"  Root Data Dir: {ROOT_DATA_DIR}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total: {total_experiments}\n")
        f.write(f"  Completed: {completed}\n")
        f.write(f"  Failed: {failed}\n")
        f.write(f"  Skipped: {skipped}\n\n")
        f.write(f"Detailed Results:\n")
        f.write(f"{'-'*80}\n")
        for log in results_log:
            f.write(f"{log}\n")
    
    print(f"Summary saved to: {summary_file}\n")
    
    # Next steps
    if completed > 0:
        print(f"{'='*80}")
        print(f"NEXT STEPS - Parse Results")
        print(f"{'='*80}")
        print(f"To compute statistics with confidence intervals, run:\n")
        for dataset in DATASETS:
            for shots in SHOTS:
                print(f"python parse_test_res.py output/{dataset.lower()}/GraphCLIP_v1/{shots}shot --ci95")
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
