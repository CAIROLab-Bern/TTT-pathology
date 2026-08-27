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


class AuxilaryMagnificationTTTModulePrecomputed(ClassifierModulePrecomputed):
    #! TODO
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.aux_head = AuxClassifier(num_classes=3,
                                           in_features=self.model.adaptor[-1].out_features*2,
                                           )

    def training_step(self, batch, batch_idx):
        x, y, y_aux = batch

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["embedding"])

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
        x, y, y_aux = batch

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["embedding"])

        loss_primary = self.loss_fn_dict["classification"](y_hat, y)*self.loss_fn_dict["weights"]["classification"]
        loss_aux = self.loss_fn_dict["auxilary"](y_aux_hat, y_aux)*self.loss_fn_dict["weights"]["auxilary"]
        if self.current_epoch < self.cfg.ttt_warmup_epochs:
            loss_aux = 0
        loss = loss_aux+loss_primary

        self.log("val_loss", loss,on_step=True,on_epoch=True)
        self.log("val_loss_aux", loss_aux,on_step=True,on_epoch=True)
        self.log("val_loss_primary", loss_primary,on_step=True,on_epoch=True)

        return {"preds": y_hat, "labels": y, "loss": loss, "aux_preds": y_aux_hat, "aux_labels": y_aux}

    def test_step(self, batch, batch_idx):
        x, y, _,_ = batch
        emb_dict = self.model.extract_feature_dict(x)
        y_hat = self.model.classifier(emb_dict["embedding"])
        
        return {"preds": y_hat, "labels": y}    

class AuxilaryMagnificationTTTModulePrecomputedTTT(ClassifierModulePrecomputed):
    #! TODO
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.aux_head = AuxClassifier(num_classes=3,
                                           in_features=self.model.adaptor[-1].out_features*2,
                                           )
        self.magnification_dict = {"0": "",
                            "1":"_x18",
                            "2":"_x22"}
        #make sure segmentation backbone is frozen

    def training_step(self, batch, batch_idx):
        x, y, y_aux = batch

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["embedding"])

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
        x, y, y_aux = batch

        emb_dict = self.model.extract_feature_dict(x)

        y_hat = self.model.classifier(emb_dict["embedding"])
        y_aux_hat = self.aux_head(emb_dict["embedding"])
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
                cls_token = temp_adapted[:,0,:]
                mean_token = torch.mean(temp_adapted[:,1:,:],dim=1)
                temp_y_aux_hat = self.aux_head(torch.cat((cls_token,mean_token),dim=1))  # predicitons using aux_head
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
        x, y, y_aux, path  = batch

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
                cls_token = temp_adapted[:,0,:]
                mean_token = torch.mean(temp_adapted[:,1:,:],dim=1)
                temp_y_aux_hat = self.aux_head(torch.cat((cls_token,mean_token),dim=1)) 
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

        return {"preds": y_hat, "labels": y, "emb_pre_TTT": emb_dict["embedding"],"emb_post_TTT": updated_emb_dict["embedding"], "preds_preTTT": self.model.classifier(emb_dict["embedding"]),"y_aux":y_aux,"y_aux_hat":temp_y_aux_hat}
