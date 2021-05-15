from modules.data import get_dataset_df, pkl_load

import random
import os
import numpy as np
import torch
from tqdm import tqdm
import sys
from pathlib import Path
import gc

cfg = {
    "version": "debug",

    "dataset_dir": "../../input/cats_dogs_dataset/train",
    "output_dir": "../../output",

    "crossval_version": "v0",
    "fold": 0,

    # "data_dir": r"data/",
    # "out_dir": r"outputs/",
    # "zarr_db_dir": r"outputs/zarr/v1/db.zarr",
    #
    # "model_version": "unet_v2",
    # "model_weights": None,
    #
    # "optimizer_version": "adam_v1",
    # "optimizer_weights": None,
    #
    # "scheduler_version": "rop_v1",
    #
    # "criterion_version": "dice_v1",
    # "dice_ths": 0.5,
    #
    # "tiles_version": "v1",
    # "augmentation_version": "v12",
    #
    # "batch_size": 8,
    #
    # "epoch_num": 1000,
    # "early_stopping": 30,
    #
    # "debug": 3,
    # "device": "cuda",
    # "n_jobs": 2,
    # "seed": 0,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, dataloader, criterion, optimizer, device="cuda", verbose="train"):
    """Train model 1 epoch. For the correct loss representation
    criterion should have reduction == 'mean'.
    """
    model.train()
    with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
        loss_epoch_sum = 0
        items_epoch = 0
        for x, _, cls in iterator:
            x, cls = x.to(device), cls[0].unsqueeze(1).to(device)
            optimizer.zero_grad()
            pred = model.forward(x)
            loss_batch = criterion(pred, cls.type_as(pred))
            loss_batch.backward()
            optimizer.step()

            loss_epoch_sum += loss_batch.item() * pred.shape[0]
            items_epoch += pred.shape[0]
            loss_epoch = loss_epoch_sum / items_epoch

            if verbose:
                log_batch = f"loss-batch: {loss_batch:.5f}"
                log_epoch = f"loss-epoch: {loss_epoch:.5f}"
                iterator.set_postfix_str(" | ".join([log_batch, log_epoch]))

    return loss_epoch


def valid(model, dataloader, criterion, device="cuda", verbose="valid"):
    """Validate model. For the correct loss representation
    criterion should have reduction == 'mean'.
    """
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            loss_epoch_sum = 0
            accuracy_epoch_sum = 0
            items_epoch = 0
            for x, _, cls in iterator:
                x, cls = x.to(device), cls[0].unsqueeze(1).to(device)
                pred = model.forward(x)
                loss_batch = criterion(pred, cls.type_as(pred))

                loss_epoch_sum += loss_batch.item() * pred.shape[0]
                accuracy_epoch_sum += ((pred > 0).type(torch.int) == cls).sum().item()
                items_epoch += pred.shape[0]

                loss_epoch = loss_epoch_sum / items_epoch
                accuracy_epoch = accuracy_epoch_sum / items_epoch

                if verbose:
                    log_batch = f"acc.-epoch: {accuracy_epoch:.5f}"
                    log_epoch = f"loss-epoch: {loss_epoch:.5f}"
                    iterator.set_postfix_str(" | ".join([log_batch, log_epoch]))

    return loss_epoch, accuracy_epoch


def main():
    results_dir = Path(cfg["output_dir"]) / "models" / "classifier" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"cross_validation_split {cfg['version']} exists!")

    seed_everything(cfg["seed"])

    dataset_df = get_dataset_df(cfg["dataset_dir"], verbose=True)
    folds_dct = pkl_load(cfg["output"] / "cross_validation_split" / cfg["crossval_version"] / "folds_dct.pkl")

    train_df = dataset_df.loc[folds_dct[cfg["fold"]]["train"]]
    valid_df = dataset_df.loc[folds_dct[cfg["fold"]]["valid"]]
    del dataset_df, folds_dct
    gc.collect()

    if cfg["version"] == "debug":
        train_df = train_df.head(30)
        valid_df = valid_df.head(30)

    # PyTorch Datasets initialization
    augmentation = get_augmentation(cfg["augmentation_version"])
    dataset = {id_: ValidDataset(db, valid_dct[id_], augmentation["valid"]) for id_ in valid_id}
    dataset["train"] = TrainDataset(db, train_df, augmentation["train"])
    dataset["train_valid"] = ValidDataset(db, train_valid_df, augmentation["valid"])

    # PyTorch DataLoaders initialization
    dataloader = {
        id_: DataLoader(dataset[id_], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["n_jobs"])
        for id_ in valid_id
    }
    dataloader["train"] = DataLoader(
        dataset["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["n_jobs"])
    dataloader["train_valid"] = DataLoader(
        dataset["train_valid"], batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["n_jobs"])

    # model
    model = get_model(cfg["model_version"])
    if cfg["model_weights"]:
        model.load_state_dict(torch.load(cfg["model_weights"]))
    model.to(cfg["device"])

    # optimizer
    optimizer = get_optimizer(cfg["optimizer_version"], model.parameters())
    if cfg["optimizer_weights"]:
        optimizer.load_state_dict(torch.load(cfg["optimizer_weights"]))

    # scheduler
    if cfg["scheduler_version"]:
        scheduler = get_scheduler(cfg["scheduler_version"], optimizer)

    # loss function
    criterion = get_criterion(cfg["criterion_version"])

    # EPOCHS
    monitor = []
    loss_min = None
    dice_max = None
    epochs_without_improvement = 0
    for epoch in range(cfg["epoch_num"]):
        print(f"Epoch-{epoch}")
        monitor_epoch = {}

        # train
        train_loss_raw = train(model, dataloader["train"], criterion, optimizer,
                               device=cfg["device"], verbose="train-train")
        monitor_epoch["train_loss_raw"] = train_loss_raw
        # train loss
        _, train_loss = valid(model, dataloader["train_valid"], criterion, dice_ths=False,
                              device=cfg["device"], verbose="train-valid")
        monitor_epoch["train_loss"] = train_loss
        # scheduler
        if cfg["scheduler_version"]:
            scheduler.step(train_loss)
        # validation
        valid_loss = 0
        valid_dice = 0
        for id_ in valid_id:
            valid_dice_id, valid_loss_id = valid(model, dataloader[id_], criterion,
                                                 dice_ths=cfg["dice_ths"], device=cfg["device"], verbose=id_)
            valid_dice += valid_dice_id
            valid_loss += valid_loss_id
        valid_loss /= len(valid_id)
        valid_dice /= len(valid_id)
        monitor_epoch["valid_loss"] = valid_loss
        monitor_epoch["valid_dice"] = valid_dice

        print(f"Train-loss: {train_loss:.5} \t Valid-loss: {valid_loss:.5} \t Valid-dice: {valid_dice:.3}")
        print("-" * 70)

        # saving progress info
        monitor.append(monitor_epoch)
        dat.pkl_dump(pd.DataFrame(monitor), results_dir / "monitor.pkl")

        # saving weights - max DICE
        if dice_max is None:
            dice_max = valid_dice
        if valid_dice >= dice_max:
            dice_max = valid_dice
            torch.save(model.state_dict(), results_dir / "model_best_dice.pt")

        # saving weights - min LOSS
        if loss_min is None:
            loss_min = valid_loss
        # loss improvement
        if valid_loss <= loss_min:
            loss_min = valid_loss
            epochs_without_improvement = 0
            # save model
            torch.save(model.state_dict(), results_dir / "model_best_loss.pt")
            torch.save(optimizer.state_dict(), results_dir / "optimizer.pt")
        # -- no loss improvement
        else:
            epochs_without_improvement += 1

        # early stopping
        if epochs_without_improvement >= cfg["early_stopping"]:
            print("EARLY STOPPING!")
            break


if __name__ == "__main__":
    main(CONFIG)
