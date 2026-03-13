import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets



# Mapping from short folder names to descriptive class names for CLIP text prompts
# NOTE: Commented out to match Tien's original setup (raw folder names used instead)
# CLASS_NAME_MAPPING = {
#     "aca_bd": "Well differentiated adenocarcinoma",
#     "aca_md": "Moderately differentiated adenocarcinoma",
#     "aca_pd": "Poorly differentiated adenocarcinoma",
#     "nor": "Normal",
#     "scc_bd": "Well differentiated squamous cell carcinoma",
#     "scc_md": "Moderately differentiated squamous cell carcinoma",
#     "scc_pd": "Poorly differentiated squamous cell carcinoma"
# }


@DATASET_REGISTRY.register()
class LungHist700(DatasetBase):
    """
    Lung Histopathology Dataset
    
    Uses Tiens's predefined few-shot splits from rebased pickle files.
    
    Directory structure:
    LungHist700/
        images/
            aca_bd/
                img1.jpg
            aca_md/
            ...
        split_fewshot_rebased/
            shot_4-seed_11111.pkl
            shot_8-seed_22222.pkl
            ...
    """

    dataset_dir = "LungHist700"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_lung_hist.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_rebased")
        mkdir_if_missing(self.split_fewshot_dir)

        # Load train/val/test from JSON split (provides the proper full test set)
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(
                f"Required split file not found: {self.split_path}\n"
                f"Please ensure you have split_lung_hist.json in the LungHist700 folder."
            )
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        print(f"Loaded split from JSON: {len(train)} train, {len(val)} val, {len(test)} test")

        # Override train/val with few-shot pickle (test stays from JSON)
        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if not os.path.exists(preprocessed):
                raise FileNotFoundError(
                    f"Required few-shot split not found: {preprocessed}\n"
                    f"Please ensure you have rebased Tien's pickle files."
                )

            print(f"Loading few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train = data["train"]
                val = data["val"]
            print(f"Few-shot override: {len(train)} train, {len(val)} val (test unchanged: {len(test)})")

        # NOTE: Class name conversion commented out to match Tien's original setup
        # train = self._convert_classnames(train)
        # val = self._convert_classnames(val)
        # test = self._convert_classnames(test)

        # Subsample classes if needed
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    
    @staticmethod
    def _convert_classnames(data_list):
        """Convert short class names to descriptive names for better CLIP text encoding."""
        converted = []
        for datum in data_list:
            old_classname = datum._classname
            # new_classname = CLASS_NAME_MAPPING.get(old_classname, old_classname)  # Commented out - Tien uses raw names
            new_classname = old_classname
            converted.append(Datum(
                impath=datum._impath,
                label=datum._label,
                classname=new_classname
            ))
        return converted