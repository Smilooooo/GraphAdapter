import pickle
import argparse
import os

# Command to look at an old pickle file:
# python view_pickle_paths.py "E:\BachelorThesis\Data\data(3)\data\LungHist700\split_fewshot_ga\shot_4-seed_11111.pkl"

# Import dassl to properly unpickle Datum objects
try:
    from dassl.data.datasets import Datum
except ImportError:
    print("[WARNING] dassl module not found, trying to unpickle anyway...")
    pass


def view_pickle_paths(pickle_file):
    """View all paths in a pickle file."""
    print(f"[INFO] Loading pickle file: {pickle_file}")
    print(f"[INFO] File size: {os.path.getsize(pickle_file)} bytes\n")
    
    with open(pickle_file, "rb") as f:
        splits = pickle.load(f)
    
    print(f"{'='*80}")
    print(f"PICKLE FILE STRUCTURE")
    print(f"{'='*80}")
    print(f"Type: {type(splits)}")
    print(f"Keys: {list(splits.keys())}\n")
    
    # Collect all paths and statistics
    all_paths = []
    split_stats = {}
    
    for split_name, data_list in splits.items():
        print(f"\n{'='*80}")
        print(f"{split_name.upper()} SPLIT - {len(data_list)} samples")
        print(f"{'='*80}\n")
        
        # Collect class statistics
        class_counts = {}
        
        for i, datum in enumerate(data_list):
            path = datum._impath
            label = datum._label
            classname = datum._classname
            
            all_paths.append(path)
            
            # Count classes
            if classname not in class_counts:
                class_counts[classname] = 0
            class_counts[classname] += 1
            
            print(f"[{i}] Label {label}: {classname}")
            print(f"    Path: {path}")
            print()
        
        split_stats[split_name] = {
            'total': len(data_list),
            'classes': class_counts
        }
    
    # Analyze paths to find common root
    print(f"\n{'='*80}")
    print(f"PATH ANALYSIS")
    print(f"{'='*80}\n")
    
    if all_paths:
        # Try to find common prefix
        common_prefix = os.path.commonpath(all_paths) if len(all_paths) > 1 else os.path.dirname(all_paths[0])
        
        print(f"Total paths: {len(all_paths)}")
        print(f"\nFirst path (full):")
        print(f"  {all_paths[0]}")
        print(f"\nLast path (full):")
        print(f"  {all_paths[-1]}")
        print(f"\nCommon prefix (suggested --rebase-from):")
        print(f"  {common_prefix}")
        
        # Find where LungHist700 appears
        for path in all_paths[:1]:
            if "LungHist700" in path or "lunghist700" in path.lower():
                idx = path.lower().find("lunghist700")
                if idx > 0:
                    suggested_root = path[:idx].rstrip('/\\')
                    print(f"\nSuggested root (before LungHist700):")
                    print(f"  {suggested_root}")
                break
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    for split_name, stats in split_stats.items():
        print(f"{split_name.upper()} Split:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Classes: {len(stats['classes'])}")
        print(f"  Samples per class:")
        for classname, count in sorted(stats['classes'].items()):
            print(f"    - {classname}: {count}")
        print()
    
    print(f"\n{'='*80}")
    print(f"USAGE EXAMPLE")
    print(f"{'='*80}\n")
    print(f"To rebase these paths, use:")
    print(f'python extract_splits.py "{pickle_file}" \\')
    print(f'    --rebase-from "<OLD_ROOT_FROM_ABOVE>" \\')
    print(f'    --rebase-to "E:/BachelorThesis/Data/data(3)/data" \\')
    print(f'    --output "<OUTPUT_PICKLE_PATH>"')


def main():
    parser = argparse.ArgumentParser(description="View paths in pickle split file")
    parser.add_argument("pickle_file", type=str, help="Path to the pickle file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pickle_file):
        print(f"[ERROR] File not found: {args.pickle_file}")
        return
    
    try:
        view_pickle_paths(args.pickle_file)
    except Exception as e:
        print(f"[ERROR] Failed to read pickle file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
