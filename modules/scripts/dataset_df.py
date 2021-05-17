from modules.data import pkl_dump

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import sys
import cv2
import json


def get_dataset_df(dataset_dir, verbose=False):
    """Collect pandas.DataFrame with *.jpg/*.txt files' names as index
    (<file_name.jpg> / <file_name.txt>) and the following columns:
        img_path - path to the *.jpg file;
        cls - image class (0 - cat; 1 - dog);
        xmin, ymin, xmax, ymax - bounding box' coordinates;
        xmin_n, ymin_n, xmax_n, ymax_n - normalized bounding box' coordinates;
        height, width - image's height and width.

    Args:
        dataset_dir (str): path to directory with images and annotation
        verbose (bool): show progress.

    Returns:
        dataset_df: pandas.DataFrame.
    """

    inf_cols = ["id", "img_path", "height", "width"]
    ann_cols = ["cls", "xmin", "ymin", "xmax", "ymax"]
    dataset_df = pd.DataFrame(columns=inf_cols + ann_cols)

    txt_files = [f for f in os.listdir(dataset_dir) if ".txt" in f]
    with tqdm(txt_files, file=sys.stdout, disable=not verbose) as iterator:
        for file in iterator:
            img_id = file.split(".")[0]

            # get image height and width
            img_path = Path(dataset_dir) / (img_id + ".jpg")
            height, width = cv2.imread(str(img_path)).shape[:2]

            # read annotation: class + bboxes
            with open(Path(dataset_dir) / file, "r") as f:
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
    dataset_df = dataset_df.set_index("id", drop=True)
    # transform cat/dog 1/2 labels to 0/1
    dataset_df["cls"] = dataset_df["cls"] - 1
    return dataset_df


def main(cfg):
    results_dir = Path(cfg["output_dir"]) / "dataset_df" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"cross_validation_split {cfg['version']} exists!")

    dataset_df = get_dataset_df(cfg["dataset_dir"], True)

    pkl_dump(dataset_df, results_dir / "dataset_df.pkl")
    with open(results_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=4)


if __name__ == "__main__":
    config = {
        "version": "train",

        "dataset_dir": r"../../input/cats_dogs_dataset/train",
        "output_dir": r"../../output",
    }
    main(config)
