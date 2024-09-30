from typing import Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision

"""A batch is a 3-tuple of imgs, labels, and metadata dict"""
BatchType = Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]

class BaseDataModuleClass(LightningDataModule):
    def __init__(self, config, shuffle: bool = True, parents=None) -> None:
        super().__init__()
        self.config = config
        self.shuffle = shuffle
        self.parents = parents
        self.train_tsfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.input_res, config.input_res)),
        torchvision.transforms.RandomHorizontalFlip(),
                                        ])
        self.val_tsfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config.input_res, config.input_res)),
        torchvision.transforms.RandomHorizontalFlip(),
                                        ])
        self.sampler = None
        self.create_datasets()

    def train_dataloader(self):
        if self.sampler is not None and self.shuffle:
            return DataLoader(
                self.dataset_train,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=False,
                batch_sampler=self.sampler,
            )
        return DataLoader(
            self.dataset_train,
            self.config.batch_size,
            shuffle=self.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    @property
    def dataset_name(self):
        raise NotImplementedError

    def create_datasets(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError