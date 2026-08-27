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
        
        self.metadata = self.metadata.sort_values("im_uuid").reset_index(drop=True)        
        self.metadata["pt_paths"] = [os.path.join(data_dir,im_id+".pt") for im_id in self.metadata["im_uuid"]]
        self.metadata["histoplus_path"] = [os.path.join(histoplus_targets,im_id+".pt") for im_id in self.metadata["im_uuid"]]
        #check if all files exist
        for fname in self.metadata["pt_paths"]:
            if not os.path.isfile(fname):
                print(f"file {fname} does not exist")

        #dataset split
        filtered_metadata = self.metadata[self.metadata["slide_id"].isin(data_split)]

        filtered_metadata = filtered_metadata.reset_index(drop=True)

        if cramers_v is not None:
            bio_classes = ['TUMOR', 'REGR_TU', 'SH_MAG', 'ADVENT', 'MUSC_PROP', 'SH_OES']
            cha_classes = list(combinations(bio_classes, 3))[bio_class_distribution_id] # for everthing where class imbalance is induced we shuffel through all 20 combinations of biological classes
            wns_classes = [x for x in bio_classes if x not in cha_classes]
            if cramers_v == 0:
                bio_combination = [("VALSET4_CHA_FULL","ADVENT" ,300),
                                   ("VALSET4_CHA_FULL","MUSC_PROP" ,300),
                                   ("VALSET4_CHA_FULL","REGR_TU" ,300),
                                   ("VALSET4_CHA_FULL", "SH_MAG",300),
                                   ("VALSET4_CHA_FULL","SH_OES",300),
                                   ("VALSET4_CHA_FULL","TUMOR" ,300),
                                   ("VALSET2_WNS","ADVENT" ,300),
                                   ("VALSET2_WNS","MUSC_PROP" ,300),
                                   ("VALSET2_WNS","REGR_TU" ,300),
                                   ("VALSET2_WNS", "SH_MAG",300),
                                   ("VALSET2_WNS","SH_OES",300),
                                   ("VALSET2_WNS","TUMOR" ,300)]
            elif cramers_v == 0.33:
                bio_combination = [("VALSET4_CHA_FULL",cha_classes[0] ,200),
                    ("VALSET4_CHA_FULL",cha_classes[1],200),
                    ("VALSET4_CHA_FULL",cha_classes[2],200),
                    ("VALSET4_CHA_FULL", wns_classes[0],400),
                    ("VALSET4_CHA_FULL",wns_classes[1],400),
                    ("VALSET4_CHA_FULL",wns_classes[2] ,400),
                    ("VALSET2_WNS",cha_classes[0],400),
                    ("VALSET2_WNS",cha_classes[1],400),
                    ("VALSET2_WNS",cha_classes[2],400),
                    ("VALSET2_WNS", wns_classes[0],200),
                    ("VALSET2_WNS",wns_classes[1],200),
                    ("VALSET2_WNS",wns_classes[2] ,200)]
            elif cramers_v == 0.67:
                bio_combination = [("VALSET4_CHA_FULL",cha_classes[0] ,100),
                    ("VALSET4_CHA_FULL",cha_classes[1],100),
                    ("VALSET4_CHA_FULL",cha_classes[2],100),
                    ("VALSET4_CHA_FULL", wns_classes[0],500),
                    ("VALSET4_CHA_FULL",wns_classes[1],500),
                    ("VALSET4_CHA_FULL",wns_classes[2] ,500),
                    ("VALSET2_WNS",cha_classes[0],500),
                    ("VALSET2_WNS",cha_classes[1],500),
                    ("VALSET2_WNS",cha_classes[2],500),
                    ("VALSET2_WNS", wns_classes[0],100),
                    ("VALSET2_WNS",wns_classes[1],100),
                    ("VALSET2_WNS",wns_classes[2] ,100)]
            elif cramers_v == 1:
                bio_combination = [("VALSET4_CHA_FULL",cha_classes[0] ,600),
                    ("VALSET4_CHA_FULL",cha_classes[1],600),
                    ("VALSET4_CHA_FULL",cha_classes[2],600),
                    ("VALSET4_CHA_FULL", wns_classes[0],0),
                    ("VALSET4_CHA_FULL",wns_classes[1],0),
                    ("VALSET4_CHA_FULL",wns_classes[2] ,0),
                    ("VALSET2_WNS",cha_classes[0],0),
                    ("VALSET2_WNS",cha_classes[1],0),
                    ("VALSET2_WNS",cha_classes[2],0),
                    ("VALSET2_WNS", wns_classes[0],600),
                    ("VALSET2_WNS",wns_classes[1],600),
                    ("VALSET2_WNS",wns_classes[2] ,600)]
            valid_ids = []
            rng = random.Random(seed) # datasplit shall produce same samples and not effect global seed 
            for med_center, bio_class, n_samples in bio_combination:
                subset = filtered_metadata[(filtered_metadata["medical_center"]==med_center) & (filtered_metadata["bio_class"]==bio_class)]
                valid_ids.extend(subset.index[rng.sample(range(len(subset)), n_samples)].tolist())
            filtered_metadata = filtered_metadata.iloc[valid_ids]

        print(pd.crosstab(filtered_metadata["bio_class"], filtered_metadata["medical_center"]))
        #set pt_paths and labels
        self.pt_paths = filtered_metadata["pt_paths"].tolist()
        self.labels_str = filtered_metadata["bio_class"]
        self.hp_paths = filtered_metadata["histoplus_path"].tolist()
        self.labels = [label_dict[x] for x in self.labels_str]
        
    def __len__(self):
        return len(self.pt_paths)
    
    def __getitem__(self, idx):
        pt = torch.load(self.pt_paths[idx], map_location="cpu").detach().squeeze()
        label = self.labels[idx] 
        hp_target = torch.load(self.hp_paths[idx],map_location="cpu")
        for key in hp_target.keys():
            hp_target[key].detach()
        return pt, label, os.path.basename(self.pt_paths[idx]), hp_target

class PatchDataModule(L.LightningDataModule):
    def __init__(self, 
                 cfg,
                 data_dir: str,
                 metadata_file: str,
                 label_dict: dict,
                 bio_class_distribution_id: int,
                 feasible_splits_dir: str,
                validation_splits_dir: str,
                histoplus_targets: str,
                num_classes: None,
                CHA_split_id: int = 0,
                CHA_split_val_id: int = 0,
                WNS_split_id: int = 0,
                WNS_split_val_id: int = 0,
                iteration: int =0,
                cramers_v: float = 0,
                name="tolkach"
                 ):
        super().__init__()
        self.cfg = cfg
        self.bio_class_distribution_id=bio_class_distribution_id
        self.histoplus_targets = histoplus_targets
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.metadata = pd.read_csv(metadata_file)
        self.cramers_v = cramers_v
        feasible_splits = pd.read_json(feasible_splits_dir)
        with open(validation_splits_dir, "r", encoding="utf-8") as f:
            validation_splits = json.load(f)
        self.data_split = {"train": feasible_splits["train"]["VALSET4_CHA_FULL"][CHA_split_id]+feasible_splits["train"]["VALSET2_WNS"][WNS_split_id],
                           "val": validation_splits["VALSET4_CHA_FULL"][CHA_split_val_id][np.random.randint(0, len(validation_splits["VALSET4_CHA_FULL"][CHA_split_val_id]))]+validation_splits["VALSET2_WNS"][WNS_split_val_id][np.random.randint(0, len(validation_splits["VALSET2_WNS"][WNS_split_val_id]))],
                           "test_id": feasible_splits["test"]["VALSET4_CHA_FULL"][CHA_split_id]+feasible_splits["test"]["VALSET2_WNS"][WNS_split_id],
                           "test_ood": self.metadata["slide_id"][self.metadata["medical_center"]=="VALSET1_UKK"]}
        self.data_split["train"] = list(set(self.data_split["train"]).difference(set(self.data_split["val"]))) #kick out cases from the training set that are in the validation split
        print(self.data_split)
    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train = PathoROBDataset(data_dir= self.data_dir,
                                          data_split=self.data_split["train"],
                                          dataset_type="train",
                                          histoplus_targets=self.histoplus_targets,
                                          label_dict=self.label_dict,
                                          metadata_file=self.metadata,
                                          bio_class_distribution_id=self.bio_class_distribution_id,
                                          cramers_v = self.cramers_v,
                                          seed = self.cfg.training.data_seed
                                          )
            self.dataset_val = PathoROBDataset(data_dir = self.data_dir,
                                          data_split=self.data_split["val"],
                                          bio_class_distribution_id=self.bio_class_distribution_id,
                                          dataset_type="val",
                                          histoplus_targets=self.histoplus_targets,
                                          label_dict=self.label_dict,
                                          metadata_file=self.metadata,
                                          seed = self.cfg.training.data_seed
                                          )
        if stage == "test":
            self.dataset_test_id = PathoROBDataset(data_dir = self.data_dir,
                                          data_split=self.data_split["test_id"],
                                          histoplus_targets=self.histoplus_targets,
                                          dataset_type="test_id",
                                          label_dict=self.label_dict,
                                          metadata_file=self.metadata,
                                          seed = self.cfg.training.data_seed
                                          )
            
            self.dataset_test_ood = PathoROBDataset(data_dir = self.data_dir,
                                data_split=self.data_split["test_ood"],
                                dataset_type="test_id",
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