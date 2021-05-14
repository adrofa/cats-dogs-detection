import sys

import pandas as pd
import numpy as np

import os
from tqdm import tqdm
import pickle
from pathlib import Path

from matplotlib import pyplot as plt

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


def pkl_dump(obj, file):
    """Dump object with pickle."""
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pkl_load(file):
    """Load object from pickle."""
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


class MyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, verbose=False):
        self.dataset_dir = dataset_dir
        self.df = self.get_dataset_df(dataset_dir, verbose)
        self.transform = A.Compose(
            [transform, ToTensorV2(p=1)],
            bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels'])
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(row["img_path"])), cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img, bboxes=[row["bboxes"]], class_labels=[row["cls"]])
        return transformed["image"], transformed["bboxes"], transformed["class_labels"]

    def get_img_bboxes(self, idx, transformed=True):
        if transformed:
            img, bboxes, _ = self.__getitem__(idx)
            img, bboxes = img.numpy().transpose(1, 2, 0), bboxes[0]
            bboxes = [int(bboxes[i] * img.shape[(i + 1) % 2])
                      for i, _ in enumerate(bboxes)]
        else:
            row = self.df.iloc[idx]
            img = cv2.cvtColor(cv2.imread(str(row["img_path"])), cv2.COLOR_BGR2RGB)
            bboxes = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

        return img, bboxes

    @staticmethod
    def get_dataset_df(dataset_dir, verbose=False):
        """Collect pandas.DataFrame with *.jpg/*.txt files' names as index
        (<file_name.jpg> / <file_name.txt>) and the following columns:
            img_path - path to the *.jpg file;
            cls - image class (1 - cat; 2 - dog);
            xmin, ymin, xmax, ymax - bounding box' coordinates;
            xmin_n, ymin_n, xmax_n, ymax_n - normalized bounding box' coordinates;
            height, width - image's height and width.

        Args:
            dataset_dir (str): path to directory with images and annotation
            verbose (str): if `str` - description of the progress bar;
                if `False` - no progress bar.

        Returns:
            dataset_df: pandas.DataFrame.
        """

        inf_cols = ["id", "img_path", "height", "width"]
        ann_cols = ["cls", "xmin", "ymin", "xmax", "ymax"]
        dataset_df = pd.DataFrame(columns=inf_cols + ann_cols)

        txt_files = [f for f in os.listdir(dataset_dir) if ".txt" in f]
        with tqdm(txt_files, desc=verbose, file=sys.stdout, disable=not verbose) as iterator:
            for file in iterator:
                img_id = file.split(".")[0]

                # get image height and width
                img_path = dataset_dir / (img_id + ".jpg")
                height, width = cv2.imread(str(img_path)).shape[:2]

                # read annotation: class + bboxes
                with open(dataset_dir / file, "r") as f:
                    annotation = map(int, f.read().split())

                # append row to df
                img_dct = {k: v for k, v in zip(ann_cols, annotation)}
                img_dct.update({
                    "id": img_id,
                    "img_path": img_path,
                    "height": height,
                    "width": width,
                })
                dataset_df = dataset_df.append(img_dct, ignore_index=True)

        # converting annotation columns type to int
        for col in ann_cols:
            dataset_df[col] = dataset_df[col].astype(int)

        # bbox normalization
        dataset_df["bboxes"] = (dataset_df[ann_cols[1:]].values /
                                dataset_df[["width", "height"] * 2].values).tolist()

        return dataset_df.set_index("id", drop=True)


def preview(img, bboxes):
    """Show an image with a bounding box from a dataset row."""
    plt.imshow(cv2.rectangle(
        img=img,
        pt1=(bboxes[0], bboxes[1]),
        pt2=(bboxes[2], bboxes[3]),
        color=(0, 255, 0),
        thickness=2
    ))
