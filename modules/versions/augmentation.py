import albumentations as A


def get_augmentation(version):
    if version == "v1":
        valid = A.Compose([
            A.Resize(128, 128, interpolation=1, always_apply=False, p=1),
            A.Normalize(mean=(0.477, 0.445, 0.395), std=(0.265, 0.26, 0.268))
            # for mean and std details check modules/scripts/image_normalization.py
        ])

        augmentation = {
            "train": valid,
            "valid": valid,
        }

    else:
        raise Exception(f"Augmentation version '{version}' is unknown!")

    return augmentation
