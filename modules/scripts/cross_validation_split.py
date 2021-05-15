from modules.data import get_dataset_df, pkl_dump

from pathlib import Path
import os
from sklearn.model_selection import KFold
import numpy as np
import json

CONFIG = {
    "version": "v0",

    "dataset_dir": r"../../input/cats_dogs_dataset/train",
    "output_dir": r"../../output",

    "n_splits": 10,
    "seed": 0,
}


def main(cfg):
    results_dir = Path(cfg["output_dir"]) / "cross_validation_split" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"cross_validation_split {cfg['version']} exists!")

    img_ids = np.array([f.split(".")[0] for f in os.listdir(cfg["dataset_dir"]) if ".jpg" in f])
    kf = KFold(n_splits=cfg["n_splits"], random_state=cfg["seed"], shuffle=True)
    folds_dct = {i: {"train": img_ids[train_idx], "valid": img_ids[valid_idx]}
                 for i, (train_idx, valid_idx) in enumerate(kf.split(img_ids))}

    pkl_dump(folds_dct, results_dir / "folds_dct.pkl")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=4)


if __name__ == "__main__":
    main(CONFIG)
