from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class DataModule(ABC):

    @abstractmethod
    def prepare_data(self) -> None:
        """Check the data is downloaded"""
        ...

    @abstractmethod
    def setup(self, stage: str) -> None:
        """Create the datasets needed for this stage"""
        ...

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        ...

