import lightning as L
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
)
from torch.utils.data import DataLoader


class DecathlonDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        task: str,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.train_data = DecathlonDataset(
            root_dir="./data",
            task=self.task,
            transform=transform,
            section="training",
            seed=self.seed,
            download=True,
        )

        self.val_data = DecathlonDataset(
            root_dir="./data",
            task=self.task,
            transform=transform,
            section="validation",
            seed=self.seed,
            download=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    data_module = DecathlonDataModule(
        root_dir="./data",
        task="Task09_Spleen",
        batch_size=4,
        num_workers=4,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print(len(train_loader), len(val_loader))
