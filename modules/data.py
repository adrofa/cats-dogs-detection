import pickle
from matplotlib import pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


def pkl_dump(obj, file):
    """Dump object with pickle.

    Args:
        obj: object to dump.
        file(str): the destination file.
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pkl_load(file):
    """Load object from pickle.

    Args:
        file(str): the pickle-file containing the object.
    """
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def preview(img, bboxes):
    """Show an image with a bounding box from a dataset row.

    Args:
        img (numpy.array): RGB image of shape HWC.
        bboxes (list): bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    plt.imshow(cv2.rectangle(
        img=img,
        pt1=(bboxes[0], bboxes[1]),
        pt2=(bboxes[2], bboxes[3]),
        color=(0, 255, 0),
        thickness=2
    ))
    plt.show()


class MyDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.df = dataset_df
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

    def get_img_bboxes(self, idx, transformed=True, normalized=False):
        """Get an image and its bounding box from the datasets.

        Args:
            idx (int): image index.
            transformed (bool): if True - get image and un-normalized bboxes as they will be passed to a network;
                if False - get image and bboxes without any transformations (from raw data).
            normalized (bool): if True - returns bboxes divided on dimension side.

        Returns:
            img (numpy.array): RGB image of shape HWC.
            bboxes (list): bounding box coordinates [xmin, ymin, xmax, ymax].
        """
        if transformed:
            img, bboxes, _ = self.__getitem__(idx)
            img, bboxes = img.numpy().transpose(1, 2, 0), bboxes[0]
            if not normalized:
                bboxes = [int(bboxes[i] * img.shape[(i + 1) % 2])
                          for i, _ in enumerate(bboxes)]
        else:
            row = self.df.iloc[idx]
            img = cv2.cvtColor(cv2.imread(str(row["img_path"])), cv2.COLOR_BGR2RGB)
            if normalized:
                bboxes = row["bboxes"]
            else:
                bboxes = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

        return img, bboxes
