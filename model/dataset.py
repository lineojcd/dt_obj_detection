import numpy as np
import torch
from PIL import Image
import os
from torchvision.transforms import functional as F

class DuckieSimDataset(object):
    def __init__(self, dir_path, splits, transforms, train=True):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.dir_path = dir_path
        self.train = train
        if self.train: # train flag set
            self.idx = splits["train"]
        else: # val
            self.idx = splits["val"]

        self.data_files = [
            os.path.join(self.dir_path, f"{i}.npz") for i in self.idx
        ]
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images, bounding boxes and labels
        data = np.load(self.data_files[idx])
        image, boxes, labels = tuple([data[f"arr_{i}"] for i in range(3)])
        num_objs = len(labels)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # variables for evaluation metrics
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.idx)