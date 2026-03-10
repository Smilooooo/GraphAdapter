import os
import pickle
import random
import math
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

NEW_CNAMES = {
    "0_N": "Normal breast tissue",
    "1_PB": "Pathological benign lesion",
    "2_UDH": "Usual ductal hyperplasia",
    "3_FEA": "Flat epithelial atypia",
    "4_ADH": "Atypical ductal hyperplasia",
    "5_DCIS": "Ductal carcinoma in situ",
    "6_IC": "Invasive carcinoma"
}


@DATASET_REGISTRY.register()
class BRACS(DatasetBase):
    """
    BRACS Dataset (Breast Cancer Histopathology)
    
    Uses predefined few-shot splits for training, with full test set.
    
    Directory structure:
    BRACS/
        split_fewshot_ga/              <- Few-shot pickle files here
            shot_4-seed_11111.pkl
        BRACS_RoI/
            latest_version/
                train/
                    0_N/, 1_PB/, ...
                test/
                    0_N/, 1_PB/, ...
    """

    dataset_dir = "BRACS/BRACS_RoI/latest_version"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.bracs_root = os.path.join(root, "BRACS")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        # split_fewshot_ga is directly under BRACS/, not under BRACS_RoI/latest_version/
        self.split_fewshot_dir = os.path.join(self.bracs_root, "split_fewshot_ga")
        
        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED
        
        if num_shots < 1:
            raise ValueError("NUM_SHOTS must be >= 1. This dataset requires few-shot splits.")
        
        # Load few-shot training samples from pickle
        preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
        
        if not os.path.exists(preprocessed):
            raise FileNotFoundError(
                f"Required few-shot split not found: {preprocessed}\n"
                f"Please ensure you have Tien's pickle files in split_fewshot_ga/"
            )
        
        print(f"\n[DEBUG] Loading few-shot data from: {preprocessed}")
        with open(preprocessed, "rb") as file:
            data = pickle.load(file)
            train_raw = data["train"]
            val_raw = data["val"]
        
        print(f"[DEBUG] Loaded {len(train_raw)} train samples and {len(val_raw)} val samples from pickle")
        
        # Rebase paths on-the-fly from Linux to Windows
        print(f"[DEBUG] Rebasing paths from Linux to Windows...")
        train = self._rebase_paths(train_raw, self.train_dir)
        val = self._rebase_paths(val_raw, self.val_dir)  # Val samples go to val/ folder
        
        # Convert class names to descriptive format
        train = self._convert_classnames(train)
        val = self._convert_classnames(val)
        
        # Load FULL test set from test/ folder (NOT from pickle - test uses ALL images in test/)
        print(f"\n[DEBUG] Loading full test set from: {self.test_dir}")
        test = self._read_test_data(self.test_dir)
        print(f"[DEBUG] Loaded {len(test)} test samples from test/ folder")
        
        print(f"\n[DEBUG] Final counts: {len(train)} train (from pickle), {len(val)} val (from pickle), {len(test)} test (from test/ folder)")

        super().__init__(train_x=train, val=val, test=test)
    
    @staticmethod
    def _rebase_paths(data_list, new_base_dir):
        """
        Rebase paths from Linux absolute paths to Windows paths on-the-fly.
        
        Similar logic to extract_splits.py but done in-memory without saving.
        
        Handles both /train/ and /val/ paths from pickle files.
        Each is mapped to its corresponding local folder based on new_base_dir.
        
        Example:
        /projects/.../BRACS/.../train/0_N/img.jpg  -> E:/.../train/0_N/img.jpg
        /projects/.../BRACS/.../val/0_N/img.jpg    -> E:/.../val/0_N/img.jpg
        """
        rebased = []
        for i, datum in enumerate(data_list):
            old_path = datum._impath
            rebased_successfully = False
            
            # Strategy: Find "/train/" or "/val/" in the path, then extract everything after it
            # Both train and val samples go to the local train/ folder
            for pattern in ["/train/", "\\train\\", "/val/", "\\val\\"]:
                if pattern in old_path:
                    # Normalize path separators to forward slash
                    normalized = old_path.replace("\\", "/")
                    
                    # Split at the pattern and take the part after (relative path)
                    # Use the forward-slash version for splitting
                    split_pattern = "/train/" if "train" in pattern else "/val/"
                    parts = normalized.split(split_pattern)
                    
                    if len(parts) == 2:
                        relative_path = parts[1]  # e.g., "0_N/img001.jpg"
                        
                        # Rebuild full path with new base directory (always train/)
                        new_path = os.path.join(new_base_dir, relative_path)
                        new_path = os.path.normpath(new_path)
                        
                        if i < 3:  # Debug first 3 paths
                            print(f"[DEBUG] Rebasing path {i}:")
                            print(f"[DEBUG]   Old: {old_path}")
                            print(f"[DEBUG]   Pattern matched: {split_pattern}")
                            print(f"[DEBUG]   Relative: {relative_path}")
                            print(f"[DEBUG]   New: {new_path}")
                        
                        rebased.append(Datum(
                            impath=new_path,
                            label=datum._label,
                            classname=datum._classname
                        ))
                        rebased_successfully = True
                        break  # Successfully rebased, move to next datum
            
            # Fallback: if pattern not found, warn and keep original
            if not rebased_successfully:
                if i < 5:  # Only warn for first 5 failures
                    print(f"[WARNING] Could not rebase path {i}: {old_path}")
                rebased.append(datum)
        
        return rebased
    
    @staticmethod
    def _convert_classnames(data_list):
        """Convert short class names to descriptive names."""
        converted = []
        for datum in data_list:
            old_classname = datum._classname
            new_classname = NEW_CNAMES.get(old_classname, old_classname)
            
            converted.append(Datum(
                impath=datum._impath,
                label=datum._label,
                classname=new_classname
            ))
        return converted
    
    def _read_test_data(self, test_dir):
        """Read all images from test/ folder."""
        categories = listdir_nohidden(test_dir)
        categories = [c for c in categories if os.path.isdir(os.path.join(test_dir, c))]
        categories.sort()
        
        test = []
        for label, category in enumerate(categories):
            category_dir = os.path.join(test_dir, category)
            images = listdir_nohidden(category_dir)
            
            # Convert class name
            classname = NEW_CNAMES.get(category, category)
            
            for im in images:
                impath = os.path.join(category_dir, im)
                test.append(Datum(impath=impath, label=label, classname=classname))
        
        return test

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.6, p_val=0.2, ignored=[], new_cnames=None):
        """
        Read images organized by class folders and split into train/val/test
        
        Expected structure:
        image_dir/  (points to BRACS/BRACS_RoI/latest_version/train/)
            0_N/
                img1.jpg
                img2.jpg
            1_PB/
                ...
        """
        
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            
            # Skip if it's not a directory
            if not os.path.isdir(category_dir):
                print(f"Skipping non-directory item: {category_dir}")
                continue
            
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            
            assert n_train > 0 and n_val > 0 and n_test > 0, \
                f"Not enough images in {category}: {n_total} total"

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test
