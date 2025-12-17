import os
import pickle
import random
import math
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

NEW_CNAMES = {
    "aca_bd": "Well differentiated adenocarcinoma",
    "aca_md": "Moderately differentiated adenocarcinoma",
    "aca_pd": "Poorly differentiated adenocarcinoma",
    "nor": "Normal",
    "scc_bd": "Well differentiated squamous cell carcinoma",
    "scc_md": "Moderately differentiated squamous cell carcinoma",
    "scc_pd": "Poorly differentiated squamous cell carcinoma"
}

@DATASET_REGISTRY.register()
class LungHist700(DatasetBase):
    """
    Lung Histopathology Dataset
    
    Directory structure:
    LungHist700/
        images/
            aca_bd/
                img1.jpg
            aca_md/
            ...
        split_fewshot_ga/
            shot_1-seed_1.pkl
            ...
    """

    dataset_dir = "LungHist700"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_LungHist700.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Load or create train/val/test splits
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            # Read and split the data from class folders inside images/
            train, val, test = self.read_and_split_data(
                self.image_dir,
                p_trn=0.6,
                p_val=0.2,
                ignored=[],
                new_cnames=NEW_CNAMES
            )
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # Handle few-shot learning - generate new splits
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Subsample classes if needed
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.6, p_val=0.2, ignored=[], new_cnames=None):
        """
        Read images organized by class folders and split into train/val/test
        
        Expected structure:
        image_dir/  (points to lungHist700/images/)
            aca_bd/
                img1.jpg
                img2.jpg
            aca_md/
                ...
        """
        # listdir_nohidden skips hidden files/folders
        categories = listdir_nohidden(image_dir)
        # filtering ignored categories
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        # calculate test proportion
        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            """
            Collate image paths into a list of Datum objects.
            This helper function takes image paths and their associated labels and class names,
            and creates a list of Datum objects for dataset processing.
            :param ims: List of image file paths or image identifiers
            :type ims: list
            :param y: Label or target value associated with the images
            :type y: int or str
            :param c: Class name corresponding to the label
            :type c: str
            :return: List of Datum objects containing image path, label, and class name
            :rtype: list[Datum]
            """
            
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