import albumentations as A

# normalization = A.Normalize(mean=(0.477, 0.445, 0.395), std=(0.265, 0.26, 0.268), p=1)
# for mean and std details check modules/scripts/image_normalization.py


def get_augmentation(version):
    if version == "v1":
        size = 448  # YOLOv2 size
        normalization = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)

        augmentation = {
            "train": A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.25),
                A.Blur(blur_limit=4, always_apply=False, p=0.25),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.25),

                A.OneOf([
                    A.RandomSizedBBoxSafeCrop(size, size, erosion_rate=0.0,
                                              interpolation=1, always_apply=False, p=0.5),
                    A.Resize(size, size, interpolation=1, always_apply=False, p=0.5),
                ], p=1),

                normalization
            ]),

            "valid": A.Compose([
                A.Resize(size, size, interpolation=1, always_apply=False, p=1),
                normalization
            ]),
        }

    else:
        raise Exception(f"Augmentation version '{version}' is unknown!")

    return augmentation
