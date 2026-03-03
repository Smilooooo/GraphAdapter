"""
Ensemble Training and Evaluation for GraphAdapter with Multiple Backbones

This script:
1. Trains GraphAdapter with each backbone (BiomedCLIP, PLIP, CONCH) on the SAME data split
2. Evaluates each model individually
3. Evaluates the ensemble (logit averaging)

This ensures fair comparison since all models see identical training data.

Usage:
    $env:HF_TOKEN = "your_token"; python ensemble_train_eval.py \
        --root "E:\BachelorThesis\Data\data(3)\data" \
        --dataset-config-file configs/datasets/lunghist700.yaml \
        --num-shots 4 \
        --seed 1 \
        --output-dir output/lunghist700/ensemble_4shot
"""

import os
import sys
import argparse
import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager

# Import custom modules to register them
import datasets.lungHist700
import datasets.bracs
import datasets.iciar2018
import trainers.baseclip_graph_v2

from trainers.model_registry import load_clip_model, MODEL_CONFIGS


# =============================================================================
# Configuration
# =============================================================================

BACKBONE_CONFIGS = {
    "biomedclip": "configs/trainers/GraphCLIP_v2/biomedclip.yaml",
    "plip": "configs/trainers/GraphCLIP_v2/plip.yaml",
    "conch": "configs/trainers/GraphCLIP_v2/conch.yaml",
}


def extend_cfg(cfg):
    """Add custom config options (same as train.py)."""
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    # Note: GRAPHADAPTER.ALPHA and BETA config values exist but are unused
    # (GraphLearner hardcodes alpha=0.1, alpha_it=0.6, beta_it=0.7)


def setup_cfg_for_backbone(args, backbone_name):
    """Setup configuration for a specific backbone."""
    cfg = get_cfg_default()
    extend_cfg(cfg)
    
    # Load dataset config
    cfg.merge_from_file(args.dataset_config_file)
    
    # Load backbone-specific config
    config_file = BACKBONE_CONFIGS[backbone_name]
    cfg.merge_from_file(config_file)
    
    # Override with command line args
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = os.path.join(args.output_dir, backbone_name)
    cfg.TRAINER.NAME = "GraphCLIP_v2"
    
    cfg.freeze()
    return cfg


def train_single_model(args, backbone_name):
    """Train a single GraphAdapter model."""
    print(f"\n{'='*60}")
    print(f"Training GraphAdapter with {backbone_name.upper()} backbone")
    print(f"{'='*60}")
    
    cfg = setup_cfg_for_backbone(args, backbone_name)
    
    # Set random seed (ensures same data split for all backbones)
    print(f"Setting fixed seed: {cfg.SEED}")
    set_random_seed(cfg.SEED)
    
    # Setup logger
    setup_logger(cfg.OUTPUT_DIR)
    
    # Build and train
    trainer = build_trainer(cfg)
    trainer.train()
    
    # Get info before deleting trainer
    best_epoch = trainer.best_epoch if hasattr(trainer, 'best_epoch') else cfg.OPTIM.MAX_EPOCH
    model_path = os.path.join(cfg.OUTPUT_DIR, "graph_learner", f"model.pth.tar-{best_epoch}")
    
    result = {
        "backbone": backbone_name,
        "cfg": cfg,
        "model_path": model_path,
        "best_epoch": best_epoch,
        "output_dir": cfg.OUTPUT_DIR
    }
    
    # Explicitly delete trainer to free memory
    del trainer
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()
    
    print(f"\n[SUCCESS] {backbone_name.upper()} training complete, memory freed")
    
    return result


def evaluate_ensemble(trained_models, args):
    """
    Evaluate ensemble of trained models using majority voting.
    
    Reloads models from checkpoints to avoid keeping all models in RAM during training.
    
    Args:
        trained_models: List of dicts with cfg and model_path for each model
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print("Evaluating Ensemble")
    print(f"{'='*60}")
    print("Reloading all models for ensemble evaluation...")
    
    # Build fresh trainers just for evaluation (to get test loader and model architecture)
    trainers = []
    for tm in trained_models:
        print(f"  Loading {tm['backbone']}...")
        # Rebuild trainer to get test loader and model structure
        trainer = build_trainer(tm["cfg"])
        # Load trained weights
        trainer.load_model(tm["output_dir"], epoch=tm["best_epoch"])
        trainer.model.eval()
        trainers.append(trainer)
    
    # Get models and their test loaders
    # IMPORTANT: Each model needs its OWN test loader (different image sizes/normalizations)
    models = [t.model for t in trainers]
    test_loaders = [t.test_loader for t in trainers]
    
    # Verify all test sets have same samples (same seed = same split)
    # Use first loader to get sample count and labels
    num_samples = len(trainers[0].test_loader.dataset)
    print(f"  Test samples: {num_samples}")
    
    # Ensemble inference - evaluate each model with its own test loader
    all_ensemble_preds = []
    all_labels = []
    individual_preds = {tm["backbone"]: [] for tm in trained_models}
    
    # First, evaluate each model individually with its correct test loader
    model_all_preds = {tm["backbone"]: [] for tm in trained_models}
    model_all_labels = {tm["backbone"]: [] for tm in trained_models}
    
    for trainer, tm in zip(trainers, trained_models):
        print(f"  Evaluating {tm['backbone']} with its test loader...")
        with torch.no_grad():
            for batch in trainer.test_loader:
                images = batch["img"].cuda()
                labels = batch["label"].cuda()
                logits = trainer.model(images)
                preds = logits.argmax(dim=1)
                model_all_preds[tm["backbone"]].append(preds.cpu())
                model_all_labels[tm["backbone"]].append(labels.cpu())
        
        model_all_preds[tm["backbone"]] = torch.cat(model_all_preds[tm["backbone"]])
        model_all_labels[tm["backbone"]] = torch.cat(model_all_labels[tm["backbone"]])
    
    # Verify all models evaluated same samples (labels should match)
    reference_labels = model_all_labels[trained_models[0]["backbone"]]
    for tm in trained_models[1:]:
        if not torch.equal(model_all_labels[tm["backbone"]], reference_labels):
            print(f"  WARNING: Label mismatch for {tm['backbone']}!")
    
    all_labels = reference_labels
    
    # Now do majority voting
    print("  Computing majority voting...")
    all_preds_stacked = torch.stack([model_all_preds[tm["backbone"]] for tm in trained_models], dim=0)
    
    # For each sample, find the most common prediction
    all_ensemble_preds = []
    for i in range(all_preds_stacked.shape[1]):
        sample_votes = all_preds_stacked[:, i]
        unique, counts = torch.unique(sample_votes, return_counts=True)
        majority_class = unique[counts.argmax()]
        all_ensemble_preds.append(majority_class)
    
    all_ensemble_preds = torch.stack(all_ensemble_preds)
    individual_preds = model_all_preds
    
    # Compute accuracies
    ensemble_acc = (all_ensemble_preds == all_labels).float().mean().item() * 100
    
    # Individual accuracies
    individual_accs = {}
    for backbone, preds in individual_preds.items():
        acc = (preds == all_labels).float().mean().item() * 100
        individual_accs[backbone] = acc
    
    return ensemble_acc, individual_accs


def main(args):
    print(f"\n{'#'*60}")
    print("# GraphAdapter Ensemble Training and Evaluation")
    print(f"# Seed: {args.seed} (ensures identical data splits)")
    print(f"# Shots: {args.num_shots}")
    print(f"# Backbones: {', '.join(args.backbones)}")
    print(f"{'#'*60}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train all models
    trained_models = []
    for i, backbone in enumerate(args.backbones, 1):
        print(f"\n[{i}/{len(args.backbones)}] Starting {backbone}...")
        result = train_single_model(args, backbone)
        trained_models.append(result)
        
        # Aggressive memory cleanup between models
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[{i}/{len(args.backbones)}] Completed. Ready for next model.")
    
    print(f"\nAll {len(args.backbones)} models trained successfully!")
    print("Preparing for ensemble evaluation...")
    
    # Evaluate ensemble
    ensemble_acc, individual_accs = evaluate_ensemble(trained_models, args)
    
    # Print and save results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nIndividual Model Accuracies:")
    for backbone, acc in individual_accs.items():
        print(f"  {backbone:15s}: {acc:.2f}%")
    
    print(f"\nEnsemble Accuracy (majority voting): {ensemble_acc:.2f}%")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, "ensemble_results.txt")
    with open(results_path, "w") as f:
        f.write(f"GraphAdapter Ensemble Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Shots: {args.num_shots}\n")
        f.write(f"Dataset: {args.dataset_config_file}\n")
        f.write(f"\nIndividual Accuracies:\n")
        for backbone, acc in individual_accs.items():
            f.write(f"  {backbone}: {acc:.2f}%\n")
        f.write(f"\nEnsemble Accuracy (majority voting): {ensemble_acc:.2f}%\n")
    
    print(f"\nResults saved to: {results_path}")
    
    return ensemble_acc, individual_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GraphAdapter ensemble")
    parser.add_argument("--root", type=str, required=True, 
                        help="Path to dataset root")
    parser.add_argument("--dataset-config-file", type=str, required=True,
                        help="Path to dataset config (e.g., configs/datasets/lunghist700.yaml)")
    parser.add_argument("--num-shots", type=int, default=4,
                        help="Number of shots per class (default: 4)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (ensures same data split for all models)")
    parser.add_argument("--output-dir", type=str, default="output/ensemble",
                        help="Output directory for all models and results")
    parser.add_argument("--backbones", type=str, nargs="+", 
                        default=["biomedclip", "plip", "conch"],
                        help="Backbones to use (default: biomedclip plip conch)")
    
    args = parser.parse_args()
    
    # Validate backbones
    for b in args.backbones:
        if b not in BACKBONE_CONFIGS:
            print(f"Error: Unknown backbone '{b}'. Available: {list(BACKBONE_CONFIGS.keys())}")
            sys.exit(1)
    
    main(args)
