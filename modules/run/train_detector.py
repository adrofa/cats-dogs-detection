from modules.data import pkl_load, MyDataset, pkl_dump
from modules.run.versions.augmentation import get_augmentation
from modules.run.versions.model import get_model
from modules.run.versions.criterion import get_criterion
from modules.run.versions.optimizer import get_optimizer
from modules.run.versions.scheduler import get_scheduler


import random
import os
import numpy as np
import torch
from tqdm import tqdm
import sys
from pathlib import Path
import gc
from torch.utils.data import DataLoader
import pandas as pd
import json
from matplotlib import pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def IoU(bboxes_pred, bboxes):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = bboxes_pred.T
    xmin_true, ymin_true, xmax_true, ymax_true = bboxes.T

    xmin_intersection = torch.stack((xmin_pred, xmin_true)).max(axis=0)[0]
    ymin_intersection = torch.stack((ymin_pred, ymin_true)).max(axis=0)[0]
    xmax_intersection = torch.stack((xmax_pred, xmax_true)).min(axis=0)[0]
    ymax_intersection = torch.stack((ymax_pred, ymax_true)).min(axis=0)[0]

    intersection_area = (xmax_intersection - xmin_intersection) * (ymax_intersection - ymin_intersection)
    intersection_area = intersection_area * (intersection_area > 0)

    true_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    union_area = true_area + pred_area - intersection_area

    iou = intersection_area / union_area

    return iou


def train(model, dataloader, criterion, optimizer, device="cuda", verbose="train"):
    """Train model 1 epoch.

    Returns:
        progress_dct (dict): dct with
            (1) cls - class predictions loss;
            (2) bbx - bounding boxes predictions loss;
            (3) weighted loss
    """
    model.train()
    with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
        progress = {
            "cls": 0,
            "bbx": 0,
            "loss": 0,
        }
        items_epoch = 0
        for x, bboxes, cls in iterator:
            x, bboxes, cls = x.to(device), torch.stack(bboxes[0]).T.to(device), cls[0].unsqueeze(1).to(device)
            optimizer.zero_grad()
            pred = model.forward(x)
            cls_pred, bboxes_pred = pred[:, 0].unsqueeze(1), pred[:, 1:]
            loss_batch, loss_cls, loss_bboxes = criterion(
                cls_pred=cls_pred,
                bboxes_pred=bboxes_pred,
                cls_true=cls.type_as(cls_pred),
                bboxes_true=bboxes.type_as(bboxes_pred),
            )
            loss_batch.backward()
            optimizer.step()

            progress["cls"] += loss_cls.item() * pred.shape[0]
            progress["bbx"] += loss_bboxes.item() * pred.shape[0]
            progress["loss"] += loss_batch.item() * pred.shape[0]
            items_epoch += pred.shape[0]

            if verbose:
                iterator.set_postfix_str(" | ".join(
                    [f"{i}: {(progress[i] / items_epoch):.5f}" for i in progress]
                ))

    progress_dct = {i: progress[i] / items_epoch for i in progress}
    return progress_dct


def valid(model, dataloader, criterion, pred_ths=0, device="cuda", verbose="valid"):
    """Validate model."""
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            progress = {
                "acc": 0,
                "iou": 0,
                "cls": 0,
                "bbx": 0,
                "loss": 0,
            }
            items_epoch = 0
            for x, bboxes, cls in iterator:
                x, bboxes, cls = x.to(device), torch.stack(bboxes[0]).T.to(device), cls[0].unsqueeze(1).to(device)
                pred = model.forward(x)
                cls_pred, bboxes_pred = pred[:, 0].unsqueeze(1), pred[:, 1:]
                loss_batch, loss_cls, loss_bboxes = criterion(
                    cls_pred,
                    bboxes_pred,
                    cls.type_as(cls_pred),
                    bboxes.type_as(bboxes_pred),
                )

                progress["acc"] += ((cls_pred > pred_ths) == cls).sum().item()
                progress["iou"] += IoU(bboxes_pred, bboxes).sum().item()
                progress["cls"] += loss_cls.item() * pred.shape[0]
                progress["bbx"] += loss_bboxes.item() * pred.shape[0]
                progress["loss"] += loss_batch.item() * pred.shape[0]
                items_epoch += pred.shape[0]

                if verbose:
                    iterator.set_postfix_str(" | ".join(
                        [f"{i}: {(progress[i] / items_epoch):.5f}"
                         for i in ["acc", "iou", "loss"]]
                    ))

    progress_dct = {i: progress[i] / items_epoch for i in progress}
    return progress_dct


def progress_chart(progress_df, chart_path):
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # loss
    axs[0, 0].set_title('Loss')
    axs[0, 0].plot(progress_df["train_loss"], label="train")
    axs[0, 0].plot(progress_df["valid_loss"], label="valid")
    axs[0, 0].legend()

    # Class loss
    axs[0, 1].set_title('Class Loss')
    axs[0, 1].plot(progress_df["train_cls"], label="train")
    axs[0, 1].plot(progress_df["valid_cls"], label="valid")
    axs[0, 1].legend()

    # BBoxes loss
    axs[0, 2].set_title('BBoxes Loss')
    axs[0, 2].plot(progress_df["train_bbx"], label="train")
    axs[0, 2].plot(progress_df["valid_bbx"], label="valid")
    axs[0, 2].legend()

    # Accuracy
    axs[1, 1].set_title('Accuracy')
    axs[1, 1].plot(progress_df["train_acc"], label="train")
    axs[1, 1].plot(progress_df["valid_acc"], label="valid")
    axs[1, 1].legend()

    # IoU
    axs[1, 2].set_title('IoU')
    axs[1, 2].plot(progress_df["train_iou"], label="train")
    axs[1, 2].plot(progress_df["valid_iou"], label="valid")
    axs[1, 2].legend()

    plt.savefig(chart_path)
    plt.close()


def main(cfg):
    results_dir = Path(cfg["output_dir"]) / "models" / "detector" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"{cfg['version']} exists!")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=4)

    seed_everything(cfg["seed"])

    # dataset_df
    dataset_df = pkl_load(cfg["dataset_df_path"])
    dataset_df["img_path"] = (dataset_df["split"] + "/" + (dataset_df.index + ".jpg")).apply(
        lambda x: Path(cfg["dataset_dir"]) / x)

    # cross-validation split
    folds_dct = pkl_load(
        Path(cfg["output_dir"]) / "cross_validation_split" / cfg["crossval_version"] / "folds_dct.pkl")
    train_df = dataset_df.loc[folds_dct[cfg["fold"]]["train"]].copy()
    valid_df = dataset_df.loc[folds_dct[cfg["fold"]]["valid"]].copy()

    # DEBUG: shorten number of images
    if cfg["debug"]:
        train_df = train_df.head(cfg["debug"])
        valid_df = valid_df.head(cfg["debug"])
    del dataset_df, folds_dct
    gc.collect()

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = {
        "train": MyDataset(train_df, augmentation["train"]),
        "train_valid": MyDataset(train_df, augmentation["valid"]),
        "valid": MyDataset(valid_df, augmentation["valid"])
    }

    # PyTorch DataLoaders initialization
    dataloader = {
        "train": DataLoader(
            dataset["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["n_jobs"]
        ),
        "train_valid": DataLoader(
            dataset["train_valid"], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"]
        ),
        "valid": DataLoader(
            dataset["valid"], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"]
        ),
    }

    # model
    model = get_model(cfg["model_version"], cfg["model_weights"], True)
    model.to(cfg["device"])

    # optimizer
    optimizer = get_optimizer(cfg["optimizer_version"], model.parameters(), cfg["optimizer_weights"])

    # scheduler
    if cfg["scheduler_version"]:
        scheduler = get_scheduler(cfg["scheduler_version"], optimizer)

    # loss function
    criterion = get_criterion(cfg["criterion_version"])

    # EPOCHS
    progress = []
    loss_min = None
    epochs_without_improvement = 0
    for epoch in range(cfg["epoch_num"]):
        print(f"Epoch-{epoch}", file=sys.stdout)

        # train
        progress_train = train(model, dataloader["train"], criterion, optimizer,
                               device=cfg["device"], verbose="train")
        # train loss (like on inference: w/o dropout etc.)
        progress_train_valid = valid(model, dataloader["train_valid"], criterion, pred_ths=cfg["pred_ths"],
                                     device=cfg["device"], verbose="train")
        if cfg["scheduler_version"]:
            scheduler.step(progress_train_valid["loss"])
        # validation
        progress_valid = valid(model, dataloader["valid"], criterion, pred_ths=cfg["pred_ths"],
                               device=cfg["device"], verbose="valid")
        if loss_min is None:
            loss_min = progress_valid["loss"]

        # Logs: epoch's results
        print(
            "\t".join([f"Train loss: {progress_train_valid['loss']:.5}",
                       f"Valid loss: {progress_valid['loss']:.5}",
                       f"Best valid loss: {loss_min:.5}"]),
            "\n" + "-" * 70,

            file=sys.stdout
        )

        # saving progress info
        progress_epoch = {"raw_"+i: progress_train[i] for i in progress_train}
        progress_epoch.update({"train_"+i: progress_train_valid[i] for i in progress_train_valid})
        progress_epoch.update({"valid_" + i: progress_valid[i] for i in progress_valid})
        progress.append(progress_epoch)
        pkl_dump(pd.DataFrame(progress), results_dir / "progress.pkl")
        progress_chart(pd.DataFrame(progress), results_dir / "progress.png")

        # saving model's weights
        if progress_valid["loss"] <= loss_min:
            loss_min = progress_valid["loss"]
            epochs_without_improvement = 0

            torch.save(model.state_dict(), results_dir / "model.pt")
            torch.save(optimizer.state_dict(), results_dir / "optimizer.pt")
        else:
            epochs_without_improvement += 1

        # early stopping
        if epochs_without_improvement >= cfg["early_stopping"]:
            print("EARLY STOPPING!")
            break


if __name__ == "__main__":
    config = {
        "version": "debug",
        "debug": False,

        "dataset_dir": "../../input/cats_dogs_dataset",
        "dataset_df_path": "../../output/dataset_df/dataset_df.pkl",
        "output_dir": "../../output",

        "crossval_version": "v1",
        "fold": 0,

        "augmentation_version": "v2",

        "batch_size": 32,
        "n_jobs": 2,
        "device": "cuda",

        "model_version": "v5",
        "model_weights": "/media/adrofa/Data/YandexDisk/Documents/Career/DS/test_tasks/2021_05_13_neurus/output/models/detector/v5/model.pt",

        "criterion_version": "v3",

        "optimizer_version": "adam_v3",
        "optimizer_weights": None,

        "scheduler_version": "rop_v1",

        "pred_ths": 0,

        "epoch_num": 1000,
        "early_stopping": 10,

        "seed": 0
    }
    main(config)
