import sys

import pandas as pd
import numpy as np

import os
from tqdm import tqdm
import pickle
from pathlib import Path

from matplotlib import pyplot as plt

import cv2


def pkl_dump(obj, file):
    """Dump object with pickle."""
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pkl_load(file):
    """Load object from pickle."""
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_dataset_df(dataset_dir, verbose=False):
    """Collect pandas.DataFrame with *.jpg/*.txt files' names as index
    (<file_name.jpg> / <file_name.txt>) and the following columns:
        img_path - path to the *.jpg file;
        cls - image class (1 - cat; 2 - dog);
        xmin, ymin, xmax, ymax - bounding box' coordinates;
        height, width - image's height and width.

    Args:
        dataset_dir (str): path to directory with images and annotation
        verbose (str): if `str` - description of the progress bar;
            if `False` - no progress bar.

    Returns:
         pandas.DataFrame
    """

    cols = ["id", "img_path", "height", "width",
            "cls", "xmin", "ymin", "xmax", "ymax"]
    dataset_df = pd.DataFrame(columns=cols)

    txt_files = [f for f in os.listdir(dataset_dir) if ".txt" in f]
    with tqdm(txt_files, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
        for file in iterator:
            img_id = file.split(".")[0]

            img_path = dataset_dir / (img_id + ".jpg")
            height, width = cv2.imread(str(img_path)).shape[:2]

            with open(dataset_dir / file, "r") as f:
                annotation = map(int, f.read().split())
            img_dct = {k: v for k, v in zip(cols[-5:], annotation)}

            img_dct.update({
                "id": img_id,
                "img_path": img_path,
                "height": height,
                "width": width,
            })

            dataset_df = dataset_df.append(img_dct, ignore_index=True)

    for col in cols[-7:]:
        dataset_df[col] = dataset_df[col].astype(int)
    dataset_df.set_index("id", drop=True, inplace=True)

    return dataset_df


def dataset_row_preview(row):
    """Show an image with a bounding box from a dataset row."""

    # read
    img = cv2.cvtColor(cv2.imread(str(row["img_path"])), cv2.COLOR_BGR2RGB)
    # add bounding box
    img = cv2.rectangle(
        img=img,
        pt1=(row["xmin"], row["ymin"]),
        pt2=(row["xmax"], row["ymax"]),
        color=(0, 255, 0),
        thickness=2
    )

    plt.imshow(img)
