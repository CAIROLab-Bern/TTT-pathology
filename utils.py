# callbacks
from lightning.pytorch.callbacks import Callback
import copy
from backbones import hoptimus1
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix, MulticlassF1Score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torch.nn.functional as F
import os
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageDraw, ImageFont
import cv2
import hydra
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb
import numpy as np
import re
# Registry of available models
BACKBONE_REGISTRY = {
    "hoptimus1": hoptimus1,
}

METRICS_REGISTRY = {
    "accuracy": {
        "class": Accuracy,
        "default_params": {"average": "micro",
                           "task":"multiclass"}
    },
    "f1_score": {
        "class": MulticlassF1Score,
        "default_params": {"average": "weighted"}
    },
}

class log_images_callback(Callback):
    def log_image(self, trainer, pl_module, batch, n=16):

        # Ensure n does not exceed batch size
        if n > batch[0].size(0):
            n = batch[0].size(0)

        x, y, _ = batch
        y_hat = pl_module.forward(x)

        inv_normalize = torchvision.transforms.Normalize(
                                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                            std=[1/0.229, 1/0.224, 1/0.255])
        imgs = []

        if pl_module.cfg.training.device == "mps":
            font = ImageFont.load_default(size=15)
        else:
            font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
            font = ImageFont.truetype(font_path, size=15)
            
        for img, real, pred in zip(x[0:n], y[0:n], y_hat[0:n]):
            pil = to_pil_image(inv_normalize(img).cpu().clamp(0,1))
            draw = ImageDraw.Draw(pil)
            draw.rectangle([0, 0, pil.width, 50], fill="white")
            draw.text((5, 5), f"real:{real}", fill="black",font=font)
            probs = F.softmax(pred, dim=0).cpu().numpy()
            probs = [f"{p:.2f}" for p in probs]
            draw.text((5, 25), f"pred:{probs}", fill="black",font=font)
            imgs.append(to_tensor(pil))
        grid = make_grid(imgs)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image("images", grid,global_step = trainer.current_epoch)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key="Examples", images=imgs)
        return
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_image(trainer,pl_module,batch)

class log_metrics_callback(Callback):
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        preds = outputs["preds"]  # or however you return them
        labels = outputs["labels"]

        for metric in pl_module.metrics.values():
            metric(preds, labels)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        preds = outputs["preds"]  # or however you return them
        labels = outputs["labels"]

        for metric in pl_module.metrics.values():
            metric(preds, labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, metric in pl_module.metrics.items():
            m = metric.compute()
            trainer.logger.log_metrics(
                {f"val_{name}": m},
                step=trainer.current_epoch,
            )
            pl_module.log(f"val_{name}", m,on_step=False,on_epoch=True)

            metric.reset()

    def on_test_epoch_end(self, trainer, pl_module):
        for name, metric in pl_module.metrics.items():
            value =  metric.compute()
            trainer.logger.log_metrics(
                {f"test_{pl_module.current_test_dataset}_{name}":value},
                step=trainer.current_epoch,
            )
            pl_module.log(f"test_{name}", value)
            metric.reset()

class log_aux_metrics_callback(Callback):
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):

        preds = outputs["aux_preds"]  # or however you return them
        labels = outputs["aux_labels"]
        for metric in pl_module.aux_metrics.values():
            metric(preds, labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, metric in pl_module.aux_metrics.items():
            trainer.logger.log_metrics(
                {f"val_aux_{name}": metric.compute()},
                step=trainer.current_epoch,
            )
            metric.reset()


# Log confusion matrix at the end of validation epoch

class plot_confusion_matrix_callback(Callback):
    def on_fit_start(self, trainer, pl_module):

        pl_module.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=pl_module.cfg.training.num_classes
        ).to(pl_module.device)

    def on_test_start(self, trainer, pl_module):

        pl_module.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=pl_module.cfg.training.num_classes
        ).to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        preds = outputs["preds"]
        labels = outputs["labels"]

        pl_module.confusion_matrix.update(preds, labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = pl_module.confusion_matrix.compute()
        plot_confusion_matrix(trainer, pl_module, cm,"val")
        pl_module.confusion_matrix.reset()


    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        preds = outputs["preds"]
        labels = outputs["labels"]

        pl_module.confusion_matrix.update(preds, labels)

    def on_test_epoch_end(self, trainer, pl_module):
        cm = pl_module.confusion_matrix.compute()
        plot_confusion_matrix(trainer, pl_module, cm,"test")
        pl_module.confusion_matrix.reset()

class MemoryMonitor(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = torch.cuda.memory_allocated() / 1024**3
        print(f"[Epoch {trainer.current_epoch}] GPU memory: {m:.2f} GB")


# utilities

# adapted from https://www.geeksforgeeks.org/deep-learning/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning/
def plot_confusion_matrix(trainer, pl_module, cm, stage):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm.cpu().numpy(),
                    annot=True,
                    ax=ax,
                    xticklabels=pl_module.cfg.datamodule.label_dict.keys(),
                    yticklabels=pl_module.cfg.datamodule.label_dict.keys())
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log confusion matrix to TensorBoard
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_figure(f"confusion_matrix_{stage}_{pl_module.current_test_dataset}", fig, trainer.current_epoch)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({
                f"confusion_matrix_{stage}_{pl_module.current_test_dataset}": wandb.Image(fig)
            })
        plt.close(fig)

def init_metrics(metric_list, num_classes):
    # Initialize metrics from a given list
    metrics = {}
    for metric in metric_list:
        metric_class = METRICS_REGISTRY[metric]["class"]
        default_params = METRICS_REGISTRY[metric].get("default_params", {})
        default_params["num_classes"] = num_classes  # Set num_classes dynamically if needed
        metrics[metric] = metric_class(**default_params)
    return nn.ModuleDict(metrics)

def init_backbone(backbone_name, num_classes, ttt_mode=None,precomputed=False):
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        ttt_mode: Test-time training mode (if any)
    
    Returns:
        Initialized model
    """
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Model '{backbone_name}' not found. Available: {list(BACKBONE_REGISTRY.keys())}")
    
    model_class = BACKBONE_REGISTRY[backbone_name]
    return model_class.get_backbone(num_classes=num_classes, ttt_mode=ttt_mode,precomputed=precomputed)

def get_loss_fn_dict(loss_fn_cfg, weights):
    loss_fn_dict = {"weights":weights}
    for key, value in loss_fn_cfg.items():
        loss_fn_dict[key] = hydra.utils.instantiate(value)
    print("Initialized loss functions:", loss_fn_dict)
    return loss_fn_dict

class log_segmentation_callback(Callback):
    def log_image(self, trainer, pl_module, batch, n=8):

        if n > batch[0].size(0):
            n = batch[0].size(0)

        x, _, _ = batch
        y_aux = pl_module.generate_targets(x)
        emb_dict = pl_module.model.extract_feature_dict(x)
        y_aux_hat = pl_module.aux_head(emb_dict["patch_tokens"])

        inv_normalize = torchvision.transforms.Normalize(
                                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                            std=[1/0.229, 1/0.224, 1/0.255])
            
        imgs = []
        for i in range(n):
            # ---- Row 1: real image ----
            img = inv_normalize(x[i]).cpu().clamp(0, 1)
            imgs.append(to_tensor(to_pil_image(img)))

            # ---- Row 2: real segmentation ----
            seg_real = y_aux[i, 0,:,:]
            if seg_real.ndim == 2:  # (H, W) → (1, H, W)
                seg_real = seg_real.unsqueeze(0)
            imgs.append(seg_real.repeat(3, 1, 1).cpu())

            # ---- Row 3: predicted segmentation ----
            seg_pred = y_aux_hat[i, 0,:,:]
            if seg_pred.ndim == 2:
                seg_pred = seg_pred.unsqueeze(0)
            imgs.append(seg_pred.repeat(3, 1, 1).cpu())

        grid = make_grid(imgs, nrow=3,normalize=True,scale_each=True)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image("images_ttt", grid,global_step = trainer.current_epoch)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key="images_ttt", images=[grid])
        return
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_image(trainer,pl_module,batch)


class log_segmentation_callback_precomputed(Callback):
    def log_image(self, trainer, pl_module, batch, n=8):

        if n > batch[0].size(0):
            n = batch[0].size(0)

        x, _, im_name, hp = batch
        y_aux = pl_module.generate_targets(hp)
        emb_dict = pl_module.model.extract_feature_dict(x)
        y_aux_hat = pl_module.aux_head(emb_dict["patch_tokens"])

        inv_normalize = torchvision.transforms.Normalize(
                                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                            std=[1/0.229, 1/0.224, 1/0.255])
            
        imgs = []

        for i in range(n):
            seg_real = y_aux[i,0,:,:]
            if seg_real.ndim == 2:  # (H, W) → (1, H, W)
                seg_real = seg_real.unsqueeze(0)
            imgs.append(seg_real.repeat(3, 1, 1).cpu())

            seg_pred = y_aux_hat[i, 0,:,:]
            if seg_pred.ndim == 2:
                seg_pred = seg_pred.unsqueeze(0)
            imgs.append(seg_pred.repeat(3, 1, 1).cpu())

        grid = make_grid(imgs, nrow=2,normalize=True,scale_each=True)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image("images_ttt", grid,global_step = trainer.current_epoch)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key="images_ttt", images=[grid])
        return im_name
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_image(trainer,pl_module,batch)


class log_reconstruction_callback_precomputed(Callback):
    def log_image(self, trainer, pl_module, batch, n=8):

        if n > batch[0].size(0):
            n = batch[0].size(0)

        x, y, y_aux, _ = batch
        emb_dict = pl_module.model.extract_feature_dict(x)
        y_aux_hat = pl_module.rae_decoder(pl_module.rae_adaptor((emb_dict["patch_tokens"])))

        imgs = []

        for i in range(n):
            imgs.append(y_aux[i,:,:,:].cpu())
            imgs.append(y_aux_hat[i, :,:,:].cpu())

        grid = make_grid(imgs, nrow=2,normalize=True,scale_each=True)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image("images_ttt", grid,global_step = trainer.current_epoch)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key="images_ttt", images=[grid])
        return
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.log_image(trainer,pl_module,batch)

class print_cuda_memory_callback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m = torch.cuda.memory_allocated() / 1024**3
        print(f"[End of Validation Epoch {trainer.current_epoch}] GPU memory: {m:.2f} GB")






class log_segmentation_callback_precomputed_test(Callback):
    def log_image(self, trainer, pl_module, batch, n=8):

        if n > batch[0].size(0):
            n = batch[0].size(0)

        x, _, im_name, hp = batch
        y_aux = pl_module.generate_targets(hp)
        emb_dict = pl_module.model.extract_feature_dict(x)
        y_aux_hat = pl_module.aux_head(emb_dict["patch_tokens"])

        inv_normalize = torchvision.transforms.Normalize(
                                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                                            std=[1/0.229, 1/0.224, 1/0.255])
            
        imgs = []

        for i in range(n):
            seg_real = y_aux[i,0,:,:]
            if seg_real.ndim == 2:  # (H, W) → (1, H, W)
                seg_real = seg_real.unsqueeze(0)
            imgs.append(seg_real.repeat(3, 1, 1).cpu())

            seg_pred = y_aux_hat[i, 0,:,:]
            if seg_pred.ndim == 2:
                seg_pred = seg_pred.unsqueeze(0)
            imgs.append(seg_pred.repeat(3, 1, 1).cpu())

        grid = make_grid(imgs, nrow=2,normalize=True,scale_each=True)
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image("images_ttt", grid,global_step = trainer.current_epoch)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key="images_ttt", images=[grid])
        return im_name
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_image(trainer,pl_module,batch)


class SavePredictionsCallback(Callback):
    def __init__(self, save_dir: str, filename: str = "test_predictions.pt"):
        super().__init__()
        self.save_dir = save_dir
        self.filename = filename
        self.outputs = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.outputs.append(outputs)

    def on_test_end(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)

        save_path = os.path.join(self.save_dir, self.filename)

        torch.save(self.outputs, save_path)

        print(f"[SavePredictionsCallback] Saved to: {save_path}")


def extract_epoch_from_log(log_path):
    """Read an output.log and return the epoch number from the checkpoint line."""
    pattern = re.compile(r'epoch=(\d+)')
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'Loaded model weights from the checkpoint' in line:
                    match = pattern.search(line)
                    if match:
                        return int(match.group(1))
    except FileNotFoundError:
        pass
    return None

def extract_ckpt_from_log(log_path):
    """Read an output.log and return the checkpoint path"""
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'Loaded model weights from the checkpoint at ' in line:
                    return line.split('Loaded model weights from the checkpoint at ')[1].strip()
    except FileNotFoundError:
        pass
    return None