import lightning as L
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
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
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.0, 1.0, 1.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(
                    keys=["image", "label"],
                    spatial_size=(128, 128, 128),
                ),
            ]
        )
        val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
            ]
        )
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
            ]
        )

        self.train_data = DecathlonDataset(
            root_dir="./data",
            task=self.task,
            transform=transform,
            section="training",
            seed=self.seed,
            download=True,
            cache_rate=0.0,
        )

        self.val_data = DecathlonDataset(
            root_dir="./data",
            task=self.task,
            transform=transform,
            section="validation",
            seed=self.seed,
            download=False,
            cache_rate=0.0,
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
        task="Task06_Lung",
        batch_size=1,
        num_workers=10,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print("Count", len(train_loader), len(val_loader))

    batch = next(iter(val_loader))
    x, y = batch["image"], batch["label"]
    print("Shape", x.shape, y.shape)

    from monai.networks.nets import UNet
    from monai.networks.layers import Norm

    unet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to("cuda")

    y_hat = unet(x.to("cuda"))

    print(y_hat.shape)
