import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix, MulticlassF1Score
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageDraw, ImageFont
import torchvision.transforms 
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils

class ClassifierModule(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.save_hyperparameters(
            OmegaConf.to_container(cfg=cfg, resolve=True)
            )

        self.model: nn.Module = utils.init_backbone(cfg.backbone, cfg.training.num_classes, cfg.ttt_mode, cfg.precomputed)
        self.optimizer: torch.optim = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.scheduler: torch.optim = hydra.utils.instantiate(cfg.scheduler, optimizer = self.optimizer)
        self.loss_fn_dict = utils.get_loss_fn_dict(cfg.loss_fn,cfg.loss_fn_weights)
        self.metrics = utils.init_metrics(metric_list= cfg.metrics, num_classes=cfg.training.num_classes).to(self.cfg.training.device)
        self.current_test_dataset = "id"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        self.log("train_loss", loss,on_step=True,on_epoch=True)
        return {"preds": y_hat, "labels": y, "loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        loss = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        self.log("val_loss", loss,on_step=True,on_epoch=True)

        return {"preds": y_hat, "labels": y, "loss": loss}
    
    def test_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y_hat = self(x)
        return {"preds": y_hat, "labels": y}    

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": {"scheduler":self.scheduler,"monitor":self.cfg.training.scheduler_monitor}}
    

