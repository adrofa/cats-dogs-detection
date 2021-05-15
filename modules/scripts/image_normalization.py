from pathlib import Path
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import sys


CONFIG = {
    "version": "v0",
    "dataset_dir": r"../../input/cats_dogs_dataset/train",
    "output_dir": r"../../output",
    "img_size": (512, 512),
}


def main(cfg):
    results_dir = Path(cfg["output_dir"]) / "image_normalization" / cfg["version"]
    try:
        os.makedirs(results_dir, exist_ok=True if cfg["version"] == "debug" else False)
    except:
        raise Exception(f"cross_validation_split {cfg['version']} exists!")

    img_paths = [Path(cfg["dataset_dir"]) / f for f in os.listdir(cfg["dataset_dir"]) if ".jpg" in f]
    if cfg["version"] == "debug":
        img_paths = img_paths[:30]

    # mean
    sum_ = np.zeros(shape=3)
    count = 0
    for img_path in tqdm(img_paths, file=sys.stdout, total=len(img_paths)):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, cfg["img_size"]) / 255
        sum_ += img.sum(axis=(0, 1))
        count += img.shape[0] * img.shape[1]
    mean = np.round(sum_ / count, 3)

    # std
    diff_squared = np.zeros(shape=3)
    for img_path in tqdm(img_paths, file=sys.stdout, total=len(img_paths)):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, cfg["img_size"]) / 255
        diff_squared += ((img - mean) ** 2).sum(axis=(0, 1))
    std = np.round(np.sqrt(diff_squared / count), 3)

    with open(results_dir / "normalization_params.txt", "w") as file:
        file.write(f"mean = [{', '.join(map(str, mean))}]")
        file.write("\n")
        file.write(f"std = [{', '.join(map(str, std))}]")

    with open(results_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=4)


if __name__ == "__main__":
    main(CONFIG)
