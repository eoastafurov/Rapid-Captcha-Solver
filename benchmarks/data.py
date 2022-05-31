from abc import ABC

from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union
import albumentations as A
from torchvision import transforms
import torch
import os
import cv2
import numpy as np
import torch.utils.data as data
import pytorch_lightning as pl

from transforms import TransformCollection, AlbuTransformCollection


class DataHyperParams:
    def __init__(
            self,
            num_classes: int,
            class_names: List[str],
            batch_size: int,
            img_size: int,
            train_data_path: str,
            val_data_path: str,
            test_data_path: str
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path


class DatasetImageFolder(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            transform: A.core.composition.Compose,
            data_hyper_params: DataHyperParams,
            supported_image_types=['jpg']
    ):
        super(DatasetImageFolder, self).__init__()
        self.root = root
        self.transform = transform

        self.class_names = data_hyper_params.class_names
        self.images_filepaths = []

        self.numpy_to_tensor_transfom = transforms.ToTensor()
        self._random_crop = A.Compose([
            A.augmentations.crops.transforms.RandomCrop(
                height=data_hyper_params.img_size,
                width=data_hyper_params.img_size,
                always_apply=True,
                p=1.0
            )
        ])

        self.class_weights = [0 for _ in range(len(self.class_names))]
        for idx, class_name in enumerate(self.class_names):
            for fname in os.listdir(os.path.join(self.root, class_name)):
                if fname.split('.')[-1] in supported_image_types:
                    if not fname.split('/')[-1].startswith('.'):
                        self.images_filepaths.append(
                            os.path.join(self.root, class_name, fname)
                        )
                        self.class_weights[idx] += 1
                

        self.class_names = {
            class_name: int(i)
            for class_name, i in zip(
                self.class_names,
                range(len(self.class_names))
            )
        }

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        label = self.class_names[image_filepath.split('/')[-2]]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._random_crop(image=image)['image']

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = self.numpy_to_tensor_transfom(image.astype(np.float32))
        # image = image / torch.tensor(255.0)

        return image, label


class IBotDataModule(pl.LightningDataModule, ABC):
    def __init__(
            self,
            data_hparams: DataHyperParams,
            transform_collection: Union[TransformCollection, None]
    ):
        super(IBotDataModule, self).__init__()
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.num_classes = data_hparams.num_classes
        self.class_names = data_hparams.class_names
        self.batch_size = data_hparams.batch_size
        self.img_size = data_hparams.img_size
        self.train_data_path = data_hparams.train_data_path
        self.val_data_path = data_hparams.val_data_path
        self.test_data_path = data_hparams.test_data_path
        self.data_hparams = data_hparams

        self.transform_collection = transform_collection

    def prepare_data(self):
        print('PREPARE DATA CALL')
        pass

    def get_train_transforms_(self):
        train_transform = self.transform_collection.train_transform(size_=self.img_size)
        return train_transform

    def get_val_transforms_(self):
        val_transform = self.transform_collection.validation_transform(size_=self.img_size)
        return val_transform

    def setup(self, stage: Optional[str] = None):
        print('SETUP CALL')
        train_transforms = self.get_train_transforms_()
        val_transforms = self.get_val_transforms_()

        self.train_data = DatasetImageFolder(
            root=self.train_data_path,
            transform=train_transforms,
            data_hyper_params=self.data_hparams
        )
        self.val_data = DatasetImageFolder(
            root=self.val_data_path,
            transform=val_transforms,
            data_hyper_params=self.data_hparams
        )
        self.test_data = DatasetImageFolder(
            root=self.test_data_path,
            transform=val_transforms,
            data_hyper_params=self.data_hparams
        )

    def train_dataloader(self):
        return data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def test_dataloader(self):
        return data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

    def val_dataloader(self):
        return data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())
