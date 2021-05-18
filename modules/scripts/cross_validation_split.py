from modules.data import pkl_dump, pkl_load

from pathlib import Path
import os
from sklearn.model_selection import KFold
import json


def main(cfg):
    results_dir = Path(cfg["output_dir"]) / "cross_validation_split" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"cross_validation_split {cfg['version']} exists!")

    dataset_df = pkl_load(Path(cfg["dataset_df_dir"]) / "dataset_df.pkl")
    img_ids = dataset_df[dataset_df["split"] == "train"].index.values
    kf = KFold(n_splits=cfg["n_splits"], random_state=cfg["seed"], shuffle=True)

    folds_dct = {i: {"train": img_ids[train_idx], "valid": img_ids[valid_idx]}
                 for i, (train_idx, valid_idx) in enumerate(kf.split(img_ids))}

    pkl_dump(folds_dct, results_dir / "folds_dct.pkl")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=4)


if __name__ == "__main__":
    config = {
        "version": "v1",

        "dataset_df_dir": r"../../output/dataset_df",
        "output_dir": r"../../output",

        "n_splits": 10,
        "seed": 0,
    }
    main(config)
