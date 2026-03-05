import pickle
import argparse
import os


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


def main():
    parser = argparse.ArgumentParser(description="Extract train/val splits from pickle file")
    parser.add_argument("pickle_file", type=str, help="Path to the pickle file")
    parser.add_argument("--export", type=str, default=None, help="Directory to export split lists")
    parser.add_argument("--quiet", action="store_true", help="Only export, don't print")
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Arguments received:")
    print(f"[DEBUG]   pickle_file: {args.pickle_file}")
    print(f"[DEBUG]   export: {args.export}")
    print(f"[DEBUG]   quiet: {args.quiet}")
    
    if not os.path.exists(args.pickle_file):
        print(f"Error: File not found: {args.pickle_file}")
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
