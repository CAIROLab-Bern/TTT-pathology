import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image
import glob
import os
from omegaconf import DictConfig, OmegaConf, open_dict
import pandas as pd
import hydra
import random
from itertools import combinations
import numpy as np
import json

class PathoROBDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 data_split, 
                 metadata_file,
                 label_dict: dict,
                 histoplus_targets,
                 transform=None,
                 augment=None,
                 max_dataset_size=None,
                 dataset_type="train",
                 cramers_v=None,
                 bio_class_distribution_id=0,
                 seed=42,
                 ):
        self.dataset_type = dataset_type
        self.data_split = data_split 
        self.metadata = metadata_file
        self.data_dir = data_dir
        self.metadata = self.metadata.sort_values("im_uuid").reset_index(drop=True)        
        self.metadata["pt_paths"] = [os.path.join(data_dir,im_id+".pt") for im_id in self.metadata["im_uuid"]]
        self.metadata["histoplus_path"] = [os.path.join(histoplus_targets,im_id+".pt") for im_id in self.metadata["im_uuid"]]
        self.magnification_dict = {"0": "",
                            "1":"_x18",
                            "2":"_x22"}
        #check if all files exist
        for fname in self.metadata["pt_paths"]:
            if not os.path.isfile(fname):
                print(f"file {fname} does not exist")

        #dataset split
        filtered_metadata = self.metadata[self.metadata["slide_id"].isin(data_split[dataset_type])]

        filtered_metadata = filtered_metadata.reset_index(drop=True)
        # sample patches according to cramers_v config (e.g. balance at 0 and highly imbalanced at 1)

        print(pd.crosstab(filtered_metadata["bio_class"], filtered_metadata["medical_center"]))
        #set pt_paths and labels
        self.pt_paths = filtered_metadata["pt_paths"].tolist()
        self.labels_str = filtered_metadata["bio_class"]
        self.hp_paths = filtered_metadata["histoplus_path"].tolist()
        self.labels = [label_dict[x] for x in self.labels_str]
        
    def __len__(self):
        return len(self.pt_paths)
    
    def __getitem__(self, idx):
        magnification = int(np.random.randint(0, 3, size=1, dtype=int))

        if self.dataset_type in ["test_id","test_ood"]: #return the original magnificaiton during testing
            magnification=0

        path = self.pt_paths[idx]
        path = self.data_dir+self.magnification_dict[str(magnification)]+"/"+path.split("/")[-1]

        pt = torch.load(path,map_location="cpu").detach().squeeze()
        label = self.labels[idx] 

        if self.dataset_type in ["test_id","test_ood","test"]: #return the original magnificaiton during testing
            return pt, label, magnification, path

        return pt, label, magnification

class PatchDataModule(L.LightningDataModule):
    def __init__(self, 
                 cfg,
                 data_dir: str,
                 metadata_file: str,
                 label_dict: dict,
                 feasible_splits_dir: str,
                histoplus_targets: str,
                num_classes: None,
                iteration: int =0,
                cramers_v: float = 0,
                name="camelyon"
                 ):
        super().__init__()
        self.cfg = cfg
        self.histoplus_targets = histoplus_targets
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.metadata = pd.read_csv(metadata_file)
        self.cramers_v = cramers_v
        with open(feasible_splits_dir, "r", encoding="utf-8") as f:
            feasible_splits = json.load(f)
        self.data_split = feasible_splits[str(cramers_v)][iteration]
        print(self.data_split)

    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train = PathoROBDataset(data_dir= self.data_dir,
                                          data_split=self.data_split,
                                          dataset_type="train",
                                          histoplus_targets=self.histoplus_targets,
                                          label_dict=self.label_dict,
                                          metadata_file=self.metadata,
                                          cramers_v = self.cramers_v,
                                          seed = self.cfg.training.data_seed
                                          )
            self.dataset_val = PathoROBDataset(data_dir = self.data_dir,
                                          data_split=self.data_split,
                                          dataset_type="val",
                                          histoplus_targets=self.histoplus_targets,
                                          label_dict=self.label_dict,
                                          metadata_file=self.metadata,
                                          seed = self.cfg.training.data_seed
                                          )
        if stage == "test":
            self.dataset_test_id = PathoROBDataset(data_dir = self.data_dir,
                                          data_split=self.data_split,
                                          histoplus_targets=self.histoplus_targets,
                                          dataset_type="test_id",
                                          label_dict=self.label_dict,
                                          metadata_file=self.metadata,
                                          seed = self.cfg.training.data_seed
                                          )
            
            self.dataset_test_ood = PathoROBDataset(data_dir = self.data_dir,
                                data_split=self.data_split,
                                dataset_type="test_ood",
                                histoplus_targets=self.histoplus_targets,
                                label_dict=self.label_dict,
                                metadata_file=self.metadata,
                                seed = self.cfg.training.data_seed
                                )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, 
                          batch_size=self.cfg.training.batch_size,
                          shuffle=True, 
                          num_workers=self.cfg.training.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, 
                          batch_size=self.cfg.training.batch_size,
                          shuffle=False, 
                          num_workers=self.cfg.training.num_workers,
                          persistent_workers=True
                          )
    
    def test_id_dataloader(self):
        return DataLoader(self.dataset_test_id,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.cfg.training.num_workers,
                          persistent_workers=True
                          )
    
    def test_ood_dataloader(self):
        return DataLoader(self.dataset_test_ood,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.cfg.training.num_workers,
                          persistent_workers=True
                          )