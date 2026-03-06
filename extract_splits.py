import pickle
import argparse
import os
import re
import glob


# Define new root for rebasing paths
NEW_ROOT = "E:/BachelorThesis/Data/data(3)/data"

def parse_pickle(file_path):
    """
    Parse a pickle file and extract train/val splits.
    """
    print(f"[DEBUG] Opening file: {file_path}")
    print(f"[DEBUG] File size: {os.path.getsize(file_path)} bytes")
    
    with open(file_path, "rb") as f:
        print("[DEBUG] Loading pickle data...")
        splits = pickle.load(f)
    
    print(f"[DEBUG] Pickle loaded successfully!")
    print(f"[DEBUG] Type of loaded data: {type(splits)}")
    print(f"[DEBUG] Keys found: {list(splits.keys())}")
    
    result = {}
    for split_name, data_list in splits.items():
        print(f"\n[DEBUG] Processing split: '{split_name}'")
        print(f"[DEBUG]   Number of entries: {len(data_list)}")
        
        result[split_name] = []
        for i, datum in enumerate(data_list):
            print(f"[DEBUG]   Entry {i}: type={type(datum).__name__}")
            
            # Debug: show available attributes
            if i == 0:
                attrs = [a for a in dir(datum) if not a.startswith('__')]
                print(f"[DEBUG]   Available attributes: {attrs}")
            
            # Extract data
            impath = getattr(datum, '_impath', None) or getattr(datum, 'impath', None)
            label = getattr(datum, '_label', None) or getattr(datum, 'label', None)
            classname = getattr(datum, '_classname', None) or getattr(datum, 'classname', None)
            
            if i < 3:  # Show first 3 entries in detail
                print(f"[DEBUG]     impath: {impath}")
                print(f"[DEBUG]     label: {label}")
                print(f"[DEBUG]     classname: {classname}")
            
            result[split_name].append({
                "path": impath,
                "filename": os.path.basename(impath) if impath else None,
                "label": label,
                "classname": classname
            })
    
    print(f"\n[DEBUG] Parsing complete!")
    return result


def print_splits(splits):
    """Pretty print the extracted splits."""
    for split_name, entries in splits.items():
        print(f"\n{'='*60}")
        print(f" {split_name.upper()} SPLIT ({len(entries)} samples)")
        print(f"{'='*60}")
        
        # Group by class
        by_class = {}
        for entry in entries:
            cls = entry.get("classname", "unknown")
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(entry["filename"])
        
        print(f"[DEBUG] Classes found in {split_name}: {list(by_class.keys())}")
        
        for classname, filenames in sorted(by_class.items()):
            print(f"\n  {classname} ({len(filenames)} images):")
            for fn in filenames:
                print(f"    - {fn}")


def rebase_paths(pickle_file, old_root, new_root, output_file):
    """Rebase paths in pickle file from old root to new root."""
    print(f"\n[DEBUG] Rebasing paths in pickle file")
    print(f"[DEBUG]   Old root: {old_root}")
    print(f"[DEBUG]   New root: {new_root}")
    print(f"[DEBUG]   Input:    {pickle_file}")
    print(f"[DEBUG]   Output:   {output_file}")
    
    # Load original pickle
    with open(pickle_file, "rb") as f:
        splits = pickle.load(f)
    
    # Normalize paths for comparison
    old_root_norm = os.path.normpath(old_root)
    new_root_norm = os.path.normpath(new_root)
    
    print(f"\n[DEBUG] Processing splits...")
    updated_count = 0
    
    for split_name, data_list in splits.items():
        print(f"\n[DEBUG] Processing split: '{split_name}' ({len(data_list)} entries)")
        
        for i, datum in enumerate(data_list):
            old_path = datum._impath
            old_path_norm = os.path.normpath(old_path)
            
            # Replace old root with new root
            if old_path_norm.startswith(old_root_norm):
                relative_path = os.path.relpath(old_path_norm, old_root_norm)
                new_path = os.path.join(new_root_norm, relative_path)
                datum._impath = new_path
                updated_count += 1
                
                if i < 3:  # Show first 3 updates
                    print(f"[DEBUG]   Updated path {i}:")
                    print(f"[DEBUG]     Old: {old_path}")
                    print(f"[DEBUG]     New: {new_path}")
            else:
                print(f"[WARNING] Path doesn't start with old_root: {old_path}")
    
    print(f"\n[DEBUG] Updated {updated_count} paths")
    
    # Save updated pickle
    print(f"\n[DEBUG] Saving updated pickle to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[SUCCESS] Rebased pickle saved to: {output_file}")
    return splits


def export_splits(splits, output_dir):
    """Export splits to text files for easy reference."""
    print(f"\n[DEBUG] Exporting splits to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, entries in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}_files.txt")
        print(f"[DEBUG] Writing {len(entries)} entries to: {output_file}")
        
        with open(output_file, "w") as f:
            for entry in entries:
                f.write(f"{entry['classname']}/{entry['filename']}\n")
        print(f"Exported {split_name} to: {output_file}")


def rebase_batch(input_dir, old_root, new_root, output_dir):
    """Rebase all pickle files in a directory."""
    print(f"\n[BATCH MODE] Rebasing all pickle files")
    print(f"[DEBUG]   Input dir:  {input_dir}")
    print(f"[DEBUG]   Old root:   {old_root}")
    print(f"[DEBUG]   New root:   {new_root}")
    print(f"[DEBUG]   Output dir: {output_dir}")
    
    # Find all .pkl files
    pickle_files = glob.glob(os.path.join(input_dir, "*.pkl"))
    
    if not pickle_files:
        print(f"[ERROR] No .pkl files found in {input_dir}")
        return
    
    print(f"\n[INFO] Found {len(pickle_files)} pickle files to process\n")
    
    success_count = 0
    fail_count = 0
    
    for i, pickle_file in enumerate(sorted(pickle_files), 1):
        filename = os.path.basename(pickle_file)
        output_file = os.path.join(output_dir, filename)
        
        print(f"[{i}/{len(pickle_files)}] Processing: {filename}")
        
        try:
            rebase_paths(pickle_file, old_root, new_root, output_file)
            success_count += 1
            print(f"  ✓ Success\n")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ Failed: {e}\n")
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"  Success: {success_count}/{len(pickle_files)}")
    print(f"  Failed:  {fail_count}/{len(pickle_files)}")


def main():
    parser = argparse.ArgumentParser(description="Extract train/val splits from pickle file")
    parser.add_argument("pickle_file", type=str, help="Path to pickle file or directory (for batch mode)")
    parser.add_argument("--export", type=str, default=None, help="Directory to export split lists")
    parser.add_argument("--quiet", action="store_true", help="Only export, don't print")
    parser.add_argument("--rebase-from", type=str, default=None, help="Old root path to replace")
    parser.add_argument("--rebase-to", type=str, default=None, help="New root path")
    parser.add_argument("--output", type=str, default=None, help="Output pickle file path or directory (for rebasing)")
    parser.add_argument("--batch", action="store_true", help="Batch mode: process all .pkl files in directory")
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Arguments received:")
    print(f"[DEBUG]   pickle_file: {args.pickle_file}")
    print(f"[DEBUG]   export: {args.export}")
    print(f"[DEBUG]   quiet: {args.quiet}")
    print(f"[DEBUG]   batch: {args.batch}")
    
    if not os.path.exists(args.pickle_file):
        print(f"Error: File not found: {args.pickle_file}")
        return
    
    # Check if batch rebasing is requested
    if args.batch and args.rebase_from and args.rebase_to:
        if not os.path.isdir(args.pickle_file):
            print("[ERROR] In batch mode, pickle_file must be a directory")
            return
        if not args.output:
            print("[ERROR] --output is required for batch rebasing")
            return
        
        try:
            rebase_batch(args.pickle_file, args.rebase_from, args.rebase_to, args.output)
            return
        except Exception as e:
            print(f"[ERROR] Batch rebasing failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Check if single file rebasing is requested
    if args.rebase_from and args.rebase_to:
        if not args.output:
            print("[ERROR] --output is required when rebasing paths")
            return
        
        try:
            splits = rebase_paths(args.pickle_file, args.rebase_from, args.rebase_to, args.output)
            print("\n[SUCCESS] Path rebasing complete!")
            return
        except Exception as e:
            print(f"[ERROR] Failed to rebase paths: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Parse the pickle file
    try:
        splits = parse_pickle(args.pickle_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse pickle file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    if not args.quiet:
        print_splits(splits)
    
    # Export if requested
    if args.export:
        export_splits(splits, args.export)
    
    # Summary
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    for split_name, entries in splits.items():
        classes = set(e.get("classname", "unknown") for e in entries)
        print(f"  {split_name}: {len(entries)} images across {len(classes)} classes")


if __name__ == "__main__":
    main()
