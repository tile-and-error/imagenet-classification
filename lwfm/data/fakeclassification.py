from lwfm.data.modules import DataModule
from torch.utils.data import Dataset, DataLoader
import torch

from lwfm.data.collate import classification_collate

class FakeClassificationDataset(Dataset):
    def __init__(
            self,
            seed: int,
            num_samples: int,
            num_classes: int,
            image_size: tuple[int, int]
    ):
        generator = torch.Generator().manual_seed(seed)

        self.images = torch.rand(
            num_samples,
            3,
            *image_size,
            generator=generator
        )

        self.labels = torch.randint(
            num_classes,
            (num_samples,),
            generator=generator
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "inputs": {"image": self.images[index]},
            "targets": {"class": self.labels[index]},
            "meta": {"sample_id": f"fake_{index:06d}"},
        }

class FakeClassificationDataModule(DataModule):
    def __init__(
            self,
            seed: int,
            num_classes: int,
            num_train: int,
            num_val: int,
            image_size: tuple[int, int],
            batch_size: int,
            num_workers: int = 0,
    ):
        self.num_classes = num_classes
        self.num_train = num_train
        self.num_val = num_val
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        if stage in ("fit", "all"):
            self.train_dataset = FakeClassificationDataset(
                self.seed,
                self.num_train,
                self.num_classes,
                self.image_size
            )
            self.val_dataset = FakeClassificationDataset(
                self.seed+1,
                self.num_val,
                self.num_classes,
                self.image_size
            )
        if stage in ("test", "all"):
            self.test_dataset = FakeClassificationDataset(
                self.seed+2,
                self.num_val,
                self.num_classes,
                self.image_size
            )

    def _loader(self, dataset, shuffle: bool):
        if dataset is None:
            raise RuntimeError("Call setup() before using a loader")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=classification_collate
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)
    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)
    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

