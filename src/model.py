import lightning as L
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import compute_generalized_dice
from monai.inferers import sliding_window_inference


class DecathlonModel(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 3e-4,
        use_scheduler: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        # Define the model
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        # Define the loss function
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True)

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch):
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # define inference method
    def _inference(self, input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=self,
                overlap=0.5,
            )

        VAL_AMP = True
        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        y_pred = self._inference(x)
        dice = compute_generalized_dice(y_pred, y)
        dice = dice.mean() if len(dice) > 0 else dice
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=True, on_epoch=True, prog_bar=True)

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
