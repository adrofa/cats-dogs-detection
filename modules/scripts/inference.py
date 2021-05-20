from modules.data import pkl_load, MyDataset, pkl_dump
from modules.run.versions.augmentation import get_augmentation
from modules.run.versions.model import get_model
from modules.run.versions.criterion import IoU

from pathlib import Path
import os
import json
import gc
from torch.utils.data import DataLoader
import torch
import sys
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


def predict(model, dataloader, device="cuda", verbose="inference"):
    model.eval()
    pred = []
    with torch.no_grad():
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            inference_time = []
            for x, _, _ in iterator:
                start_time = time.time() * 1000

                x = x.to(device)
                p = model.forward(x)
                pred.append(p.to("cpu").numpy())

                end_time = time.time() * 1000
                inference_time.append(end_time-start_time)

    pred = np.concatenate(pred, axis=0)
    return pred, inference_time


def get_img_bboxes(df, img_id, pred=False):
    """Get an image and its bounding box from the datasets.

    Args:
        df (pandas.DataFrame): DataFrame generated during `main` function below,
            should contain images ids as index, ground truth and predicted
            bounding boxes coordinates, images paths and IoU (per image).
        img_id (int): image id (image_id.jpg).
        pred (bool): if True - returns predicted bounding boxes.

    Returns:
        img (numpy.array): RGB image of shape HWC.
        bboxes (list): bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    row = df.loc[img_id]
    img = cv2.cvtColor(cv2.imread(str(row["img_path"])), cv2.COLOR_BGR2RGB)
    bbx_cols = ["xmin", "ymin", "xmax", "ymax"]
    if pred:
        bbx_cols = [col+"_pred" for col in bbx_cols]
    bbx = [row[col] for col in bbx_cols]
    return img, bbx


def results_chart(df, img_id_lst, chart_path=None, color="green"):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }

    n = len(img_id_lst)
    fig, axs = plt.subplots(n, 2, figsize=(8, 4 * n))
    for i in range(n):
        img_id = img_id_lst[i]
        row = df.loc[img_id]

        # ground truth
        img, bboxes = get_img_bboxes(df, img_id, pred=False)
        axs[i, 0].imshow(cv2.rectangle(
            img=img,
            pt1=(bboxes[0], bboxes[1]),
            pt2=(bboxes[2], bboxes[3]),
            color=colors["blue"],
            thickness=2
        ))
        axs[i, 0].set_title(f"{row.name} | Ground Truth")

        # predicted
        img, bboxes = get_img_bboxes(df, img_id, pred=True)
        axs[i, 1].imshow(cv2.rectangle(
            img=img,
            pt1=(bboxes[0], bboxes[1]),
            pt2=(bboxes[2], bboxes[3]),
            color=colors[color],
            thickness=2
        ))
        axs[i, 1].set_title(f"{row.name} | Prediction | IoU - {row['IoU']:.2}")

    fig.set_facecolor('w')
    plt.tight_layout()
    if chart_path:
        plt.savefig(chart_path)
    else:
        plt.show()
    plt.close()


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

    if cfg["crossval_version"] == "test":
        img_ids = dataset_df[dataset_df["split"] == "valid"].index.values
    else:
        folds_dct = pkl_load(
            Path(cfg["output_dir"]) / "cross_validation_split" / cfg["crossval_version"] / "folds_dct.pkl")
        img_ids = folds_dct[cfg["fold"]][cfg["split"]]

    df = dataset_df.loc[img_ids].copy()
    del dataset_df
    gc.collect()

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = MyDataset(df, augmentation["valid"])

    # PyTorch DataLoaders initialization
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)

    # model
    model = get_model(cfg["model_version"], cfg["model_weights"], True)
    model.to(cfg["device"])

    pred, inference_time = predict(model, dataloader, device=cfg["device"], verbose="inference")

    df["logit"] = pred[:, 0]
    df["cls_pred"] = (df["logit"] > 0).astype(int)
    df["bboxes_pred"] = pred[:, 1:].tolist()
    # un-normalize bboxes
    bboxes_cols = ["xmin", "ymin", "xmax", "ymax"]
    for i, col in enumerate([col+"_pred" for col in bboxes_cols]):
        dim_col = "width" if col[0] == "x" else "height"
        df[col] = pred[:, i + 1] * df[dim_col]

        # put bboxes into image borders
        if col[1:4] == "min":
            df.loc[df[col] < 0, col] = 0
        else:  # max
            df[col] = df[[col, dim_col]].min(axis=1)

        df[col] = df[col].astype(int)

    # IoU per Image
    df["IoU"] = IoU(bboxes_pred=torch.tensor(df[[col+"_pred" for col in bboxes_cols]].values),
                    bboxes=torch.tensor(df[bboxes_cols].values))
    # inference time
    df["time"] = inference_time

    pkl_dump(df, results_dir / "pred.pkl")

    # saving charts with TOP3 best and worst predictions
    top3_best = df.sort_values(by="IoU", ascending=False).iloc[:3].index.values
    results_chart(df, top3_best, chart_path=results_dir / "top3_best.png", color="green")
    top3_worst = df.sort_values(by="IoU", ascending=True).iloc[:3].index.values
    results_chart(df, top3_worst, chart_path=results_dir / "top3_worst.png", color="red")

    # submission
    mIoU = df['IoU'].mean()
    classification_accuracy = (df["cls"] == df["cls_pred"]).mean()
    inference_time = df["time"].mean()
    res_str = ", ".join([
        f"mIoU {round(mIoU * 100)}%",  # mIoU
        f"classification accuracy {round(classification_accuracy * 100)}%",
        f"{round(inference_time, 2)}ms",
        f"{cfg['train_size']} train",
        f"{len(df)} valid",
    ]) + "."
    print(res_str)


if __name__ == "__main__":
    config = {
        "version": "v1",

        "dataset_dir": "../../input/cats_dogs_dataset",
        "dataset_df_path": "../../output/dataset_df/dataset_df.pkl",
        "output_dir": "../../output",

        "model_version": "v6",
        "model_weights": "../../output/models/detector/v11/model.pt",

        "train_size": "2686",
        "train_size_comment": "train length of cross_validation_split-v1 fold-0",

        "crossval_version": "test",
        "fold": None,
        "split": None,

        "augmentation_version": "v1",

        "device": "cuda",

    }
    main(config)
