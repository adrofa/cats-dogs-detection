from modules.data import pkl_load, MyDataset, pkl_dump
from modules.run.versions.augmentation import get_augmentation
from modules.run.versions.model import get_model

from pathlib import Path
import os
import json
import gc
from torch.utils.data import DataLoader
import torch
import sys
from tqdm import tqdm
import numpy as np


def predict(model, dataloader, device="cuda", verbose="inference"):
    model.eval()
    pred = []
    with torch.no_grad():
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            for x, _, _ in iterator:
                x = x.to(device)
                p = model.forward(x)
                pred.append(p.to("cpu").numpy())

    pred = np.concatenate(pred, axis=0)
    return pred


def main(cfg):
    results_dir = Path(cfg["output_dir"]) / "inference" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"cross_validation_split {cfg['version']} exists!")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=4)

    # dataset_df
    dataset_df = pkl_load(cfg["dataset_df_path"])
    dataset_df["img_path"] = (dataset_df["split"] + "/" + (dataset_df.index + ".jpg")).apply(
        lambda x: Path(cfg["dataset_dir"]) / x)

    folds_dct = pkl_load(
        Path(cfg["output_dir"]) / "cross_validation_split" / cfg["crossval_version"] / "folds_dct.pkl")

    df = dataset_df.loc[folds_dct[cfg["fold"]][cfg["split"]]].copy()
    del dataset_df, folds_dct
    gc.collect()

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = MyDataset(df, augmentation["valid"])

    # PyTorch DataLoaders initialization
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"])

    # model
    model = get_model(cfg["model_version"], cfg["model_weights"], True)
    model.to(cfg["device"])

    pred = predict(model, dataloader, device=cfg["device"], verbose="inference")

    df["cls_pred"] = pred[:, 0]
    df["bboxes_pred"] = pred[:, 1:].tolist()
    for i, col in enumerate(["xmin_pred", "ymin_pred", "xmax_pred", "ymax_pred"]):
        dim_col = "width" if col[0] == "x" else "height"
        df[col] = pred[:, i + 1] * df[dim_col]

        if col[1:4] == "min":
            df.loc[df[col] < 0, col] = 0
        else:  # max
            df[col] = df[[col, dim_col]].min(axis=1)

        df[col] = df[col].astype(int)

    pkl_dump(df, results_dir / "pred.pkl")


if __name__ == "__main__":
    config = {
        "version": "debug",

        "dataset_dir": "../../input/cats_dogs_dataset",
        "dataset_df_path": "../../output/dataset_df/dataset_df.pkl",
        "output_dir": "../../output",

        "model_version": "v3",
        "model_weights": "/media/adrofa/Data/YandexDisk/Documents/Career/DS/test_tasks/2021_05_13_neurus/output/models/detector/v9/model.pt",

        "crossval_version": "v1",
        "fold": 0,
        "split": "valid",

        "augmentation_version": "v1",

        "batch_size": 32,
        "n_jobs": 2,
        "device": "cuda",

    }
    main(config)
