import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageDraw, ImageFont
import torchvision.transforms 
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pytorch_lightning_spells.losses import Poly1FocalLoss
from modules.base_module_precomputed import ClassifierModule as ClassifierModulePrecomputed
import copy
import multiprocessing
import monai.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import utils
import numpy as np


class AuxClassifier(nn.Module):
    def __init__(self,
                 in_features=1024,
                 num_classes=1
                 ):
            super().__init__()
            self.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features,
                            out_features=num_classes),
                )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classifier(x)
        return out

# transformer head - optimized with chatgpt to avoid checkerboard artifacts
class TransformerSegHead(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 hidden_channels=256,
                 num_classes=1,
                 token_h=14,
                 token_w=14,
                 out_h=224,
                 out_w=224):
        super().__init__()

        self.token_h = token_h
        self.token_w = token_w
        self.out_h = out_h
        self.out_w = out_w

        # -------- Decoder blocks (Upsample → Conv) --------
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # -------- Classifier --------
        self.conv_cls = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        x:
          - (B, token_h * token_w, C)  [ViT tokens]
          - or (B, C, token_h, token_w)
        """

        # ---- Token → feature map ----
        if x.dim() == 3:
            b, n, c = x.shape
            if n != self.token_h * self.token_w:
                raise ValueError(f"Expected {self.token_h * self.token_w} tokens, got {n}")
            x = (
                x.transpose(1, 2)
                 .contiguous()
                 .view(b, c, self.token_h, self.token_w)
            )
        elif x.dim() != 4:
            raise ValueError(f"Unsupported x shape {x.shape}")

        # ---- Decode ----
        x = self.block1(x)   # (B, hidden, 28, 28)
        x = self.block2(x)   # (B, hidden, 56, 56)
        x = self.block3(x)   # (B, hidden, 112, 112)

        x = self.conv_cls(x) # (B, num_classes, 112, 112)

        # ---- Final resize (safe, smooth) ----
        if x.shape[-2:] != (self.out_h, self.out_w):
            x = F.interpolate(
                x,
                size=(self.out_h, self.out_w),
                mode="bilinear",
                align_corners=False,
            )

        return x

class AuxilarySegmentationTTTModulePrecomputed(ClassifierModulePrecomputed):
    #! TODO
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.aux_head = TransformerSegHead(token_h=self.model.output_token_dim,
                                           token_w=self.model.output_token_dim,
                                           in_channels=self.model.adaptor[-1].out_features)
        #make sure segmentation backbone is frozen

    def generate_targets(self,hp_target):
        #transforms specific for auxbackbone x = transform(x)
        target = torch.Tensor()
        for key in self.cfg.hp_targets.keys():
            for i in self.cfg.hp_targets[key]:
                temp = hp_target[key].squeeze(1)[:,i,:,:].unsqueeze(1)
                target = torch.cat((target,temp),dim=1) if target.shape[0] != 0 else temp

        return target

    def training_step(self, batch, batch_idx):
        x, y, _, hp_target = batch
        y_aux = self.generate_targets(hp_target)

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["patch_tokens"])

        loss_primary = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        loss_aux = self.loss_fn_dict["auxilary"](y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]
        if self.current_epoch < self.cfg.ttt_warmup_epochs:
            loss_aux = 0
        loss = loss_aux+loss_primary
        
        self.log("train_loss", loss,on_step=True,on_epoch=True)
        self.log("train_loss_aux", loss_aux,on_step=True,on_epoch=True)
        self.log("train_loss_primary", loss_primary,on_step=True,on_epoch=True)

        self.log("train_loss", loss,on_step=True,on_epoch=True)
        return {"preds": y_hat, "labels": y, "loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, _, hp_target = batch
        y_aux = self.generate_targets(hp_target)

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["patch_tokens"])

        loss_primary = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        loss_aux = self.loss_fn_dict["auxilary"](y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]
        if self.current_epoch < self.cfg.ttt_warmup_epochs:
            loss_aux = 0
        loss = loss_aux+loss_primary

        self.log("val_loss", loss,on_step=True,on_epoch=True)
        self.log("val_loss_aux", loss_aux,on_step=True,on_epoch=True)
        self.log("val_loss_primary", loss_primary,on_step=True,on_epoch=True)

        return {"preds": y_hat, "labels": y, "loss": loss, "aux_preds": y_aux_hat, "aux_labels": y_aux}


class AuxilarySegmentationTTTModulePrecomputedTTT(ClassifierModulePrecomputed):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.aux_head = TransformerSegHead(token_h=self.model.output_token_dim,
                                           token_w=self.model.output_token_dim,
                                           in_channels=self.model.adaptor[-1].out_features)
        #make sure segmentation backbone is frozen

    def generate_targets(self,hp_target):
        #transforms specific for auxbackbone x = transform(x)
        target = torch.Tensor()
        for key in self.cfg.hp_targets.keys():
            for i in self.cfg.hp_targets[key]:
                temp = hp_target[key].squeeze(1)[:,i,:,:].unsqueeze(1)
                target = torch.cat((target,temp),dim=1) if target.shape[0] != 0 else temp

        return target

    
    def training_step(self, batch, batch_idx):
        x, y, _, hp_target = batch
        y_aux = self.generate_targets(hp_target)

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["patch_tokens"])

        loss_primary = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        loss_aux = self.loss_fn_dict["auxilary"](y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]
        if self.current_epoch < self.cfg.ttt_warmup_epochs:
            loss_aux = 0
        loss = loss_aux+loss_primary
        
        self.log("train_loss", loss,on_step=True,on_epoch=True)
        self.log("train_loss_aux", loss_aux,on_step=True,on_epoch=True)
        self.log("train_loss_primary", loss_primary,on_step=True,on_epoch=True)

        self.log("train_loss", loss,on_step=True,on_epoch=True)
        return {"preds": y_hat, "labels": y, "loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, _, hp = batch
        y_aux = self.generate_targets(hp)

        # Extract features and compute auxiliary loss
        emb_dict = self.model.extract_feature_dict(x)
        y_aux_hat = self.aux_head(emb_dict["patch_tokens"])
        loss_aux_original = self.loss_fn_dict["auxilary"](y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]

        # Create temporary model and perform TTT
        with torch.enable_grad():
            temp_adaptor = copy.deepcopy(self.model.adaptor)
            temp_adaptor.requires_grad_(True)
            temp_optimizer = torch.optim.SGD(
                temp_adaptor.parameters(),
                lr=self.cfg.ttt_lr
            )

            # adjusting the adaptor
            for steps in range(self.cfg.ttt_steps):
                temp_adapted = temp_adaptor(x)  # adaptor forward pass
                temp_y_aux_hat = self.aux_head(temp_adapted[:,1:,:])  # predicitons using aux_head
                temp_loss_aux = self.loss_fn_dict["auxilary"](temp_y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]
                temp_optimizer.zero_grad()
                temp_loss_aux.backward()
                temp_optimizer.step()
                
        
        if (self.cfg.ttt_online) & (self.cfg.ttt_active_val_learning):
            self.model.adaptor = temp_adaptor
            self.model.eval()
            with torch.no_grad():
                updated_emb_dict = self.model.extract_feature_dict(x)
                y_hat = self.model.classifier(updated_emb_dict["embedding"])
        else:
            temp_model = copy.deepcopy(self.model)
            temp_model.adaptor = temp_adaptor
            temp_model.eval()
            with torch.no_grad():
                updated_emb_dict = temp_model.extract_feature_dict(x)
                y_hat = self.model.classifier(updated_emb_dict["embedding"])
        
        loss_primary = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        
        if self.current_epoch < self.cfg.ttt_warmup_epochs:
            loss_aux_original = 0
        loss = loss_aux_original + loss_primary

        mean_distance_embeddings = torch.nn.functional.mse_loss(emb_dict["embedding"],updated_emb_dict["embedding"])
        self.log("mean_distance_embeddings_after_TTT",mean_distance_embeddings,on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_loss_aux", loss_aux_original, on_step=True, on_epoch=True)
        self.log("val_loss_primary", loss_primary, on_step=True, on_epoch=True)

        return {"preds": y_hat, "labels": y, "loss": loss, "aux_preds": y_aux_hat, "aux_labels": y_aux}

    def test_step(self, batch, batch_idx):
        x, y, _, hp = batch
        y_aux = self.generate_targets(hp)

        # Extract features and compute auxiliary loss
        emb_dict = self.model.extract_feature_dict(x)

        # Create temporary model and perform TTT
        with torch.enable_grad():
            temp_adaptor = copy.deepcopy(self.model.adaptor)
            temp_adaptor.requires_grad_(True)
            temp_adaptor.train()  # Set to train mode
            
            # Create temp aux_head - needs to be in train mode to propagate gradients
            temp_aux_head = copy.deepcopy(self.aux_head)
            temp_aux_head.train()  # Train mode to build computation graph
            # Don't set requires_grad - it's already copied with the right grad settings
            
            # Only optimize temp_adaptor parameters
            temp_optimizer = torch.optim.SGD(
                temp_adaptor.parameters(),  # ONLY adaptor parameters
                lr=self.cfg.ttt_lr
            )

            # adjusting the adaptor
            for steps in range(self.cfg.ttt_steps):
                temp_adapted = temp_adaptor(x)  # adaptor forward pass
                temp_y_aux_hat = temp_aux_head(temp_adapted[:,1:,:])  # Use temp_aux_head
                temp_loss_aux = self.loss_fn_dict["auxilary"](temp_y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]
                temp_optimizer.zero_grad()
                temp_loss_aux.backward()
                temp_optimizer.step()
                
        
        if self.cfg.ttt_online:
            self.model.adaptor = temp_adaptor
            self.model.eval()
            with torch.no_grad():
                updated_emb_dict = self.model.extract_feature_dict(x)
                y_hat = self.model.classifier(updated_emb_dict["embedding"])
        else:
            temp_model = copy.deepcopy(self.model)
            temp_model.adaptor = temp_adaptor
            temp_model.eval()
            with torch.no_grad():
                updated_emb_dict = temp_model.extract_feature_dict(x)
                y_hat = self.model.classifier(updated_emb_dict["embedding"])
            

        mean_distance_embeddings = torch.nn.functional.mse_loss(emb_dict["embedding"],updated_emb_dict["embedding"])
        self.log("mean_distance_embeddings_after_TTT",mean_distance_embeddings,on_step=True, on_epoch=True)

        return {"preds": y_hat, "labels": y, "emb_pre_TTT": emb_dict["embedding"],"emb_post_TTT": updated_emb_dict["embedding"], "preds_preTTT": self.model.classifier(emb_dict["embedding"])}