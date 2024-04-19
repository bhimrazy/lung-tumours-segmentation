import lightning as L
import torch
from torch import nn
from torchmetrics.functional import accuracy, dice_metric
from src.models.factory import ModelFactory
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet


class DecathlonModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        learning_rate: float = 3e-4,
        use_scheduler: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        # Define the model
        self.model = net

        # Define the loss function
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)
        # Define the metric
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.05
        )

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",  # monitor validation loss
        }

        if self.use_scheduler:
            # Add lr scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            configuration["lr_scheduler"] = scheduler

        return configuration
