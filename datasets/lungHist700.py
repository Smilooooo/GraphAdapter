import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


# Mapping from short folder names to descriptive class names for CLIP text prompts
CLASS_NAME_MAPPING = {
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
    
    Uses Tien's predefined few-shot splits for training, with full test set.
    
    Directory structure:
    LungHist700/
        images/
            aca_bd/
                img1.jpg
            aca_md/
            ...
        split_zhou_LungHist700.json  <- Full train/val/test split
        split_fewshot_rebased/
            shot_4-seed_11111.pkl    <- Few-shot training samples only
            ...
    """

    dataset_dir = "LungHist700"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_LungHist700.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_rebased")
        mkdir_if_missing(self.split_fewshot_dir)

        # Load full train/val/test split for the FULL test set
        if not os.path.exists(self.split_path):
            raise FileNotFoundError(
                f"Full split file not found: {self.split_path}\n"
                f"This file is required for the full test set."
            )
        
        _, _, test_full = OxfordPets.read_split(self.split_path, self.image_dir)
        
        # Load few-shot training samples from rebased pickle files
        num_shots = cfg.DATASET.NUM_SHOTS
        seed = cfg.SEED
        
        if num_shots < 1:
            raise ValueError("NUM_SHOTS must be >= 1. This dataset requires Tien's few-shot splits.")
        
        preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
        
        if not os.path.exists(preprocessed):
            raise FileNotFoundError(
                f"Required few-shot split not found: {preprocessed}\n"
                f"Please ensure you have rebased Tien's pickle files."
            )
        
        print(f"Loading few-shot training data from {preprocessed}")
        with open(preprocessed, "rb") as file:
            data = pickle.load(file)
            train_raw = data["train"]  # Few-shot support set for training
        
        # Convert short class names to descriptive names for better CLIP text prompts
        # e.g., "aca_bd" -> "Well differentiated adenocarcinoma"
        train = self._convert_classnames(train_raw)
        test = self._convert_classnames(test_full)  # Use FULL test set
        val = test  # Use same split for validation during training
        
        print(f"Few-shot setup: {len(train)} train samples, {len(test)} test samples")

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
            new_classname = CLASS_NAME_MAPPING.get(old_classname, old_classname)
            
            # Create new Datum with updated classname
            converted.append(Datum(
                impath=datum._impath,
                label=datum._label,
                classname=new_classname
            ))
        return converted