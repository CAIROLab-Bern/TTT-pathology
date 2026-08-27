from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint,EarlyStopping
from ttt.multitask import L
import modules.datamodule_precomputed
import modules.datamodule_precomputed_camelyon
from lightning.pytorch.callbacks import Callback
import utils
import torch
import wandb
import pandas as pd
import json 

@hydra.main(version_base=None, config_path="./configs", config_name="segmentation_config_ubelix")
def main(cfg: DictConfig):
    print(os.getcwd())
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({'hydra': {'run': {'dir': HydraConfig.get().run.dir}}})
        )
    OmegaConf.set_struct(cfg, True)
    print(OmegaConf.to_yaml(cfg))

    #set seed for reproducibility
    L.seed_everything(cfg.training.seed, workers=True)
    feasible_splits = pd.read_json(cfg.datamodule.feasible_splits_dir) #get all feasbile case-level splits of WNS & CHA

    if cfg.datamodule.name == "tolkach":
        # generate the indices used for train/val/test_id

        with open(cfg.datamodule.validation_splits_dir, "r", encoding="utf-8") as f: #get corresponding cases for the validation set for each split
                validation_splits = json.load(f)

        cha_splits = torch.randperm(len(feasible_splits["train"]["VALSET4_CHA_FULL"]))[:cfg.training.iterations_per_model]
        if cfg.training.iterations_per_model < 17:
            wns_splits = torch.randperm(len(feasible_splits["train"]["VALSET2_WNS"]))[:cfg.training.iterations_per_model]
        else:
            wns_splits = list(range(17))
            wns_splits.extend(torch.randperm(len(feasible_splits["train"]["VALSET2_WNS"]))[:cfg.training.iterations_per_model-17])

        cha_splits_val = []
        wns_splits_val = []
        for i in range(cfg.training.iterations_per_model):
            cha_splits_val.append(torch.randperm(len(validation_splits["VALSET4_CHA_FULL"][cha_splits[i]]))[0])
            wns_splits_val.append(torch.randperm(len(validation_splits["VALSET2_WNS"][wns_splits[i]]))[0])
        
        print("CHA split ids:")
        print(cha_splits)
        print(cha_splits_val)
        print("WNS split ids:")
        print(wns_splits)
        print(wns_splits_val)
    results = dict()
    for i in range(cfg.training.iterations_per_model):
        filepath =os.path.join(cfg.hydra.run.dir, 'logs',str(i),"wandb","latest-run","files","wandb-summary.json") #check if run already exists and contains results, if yes skip it (useful when training is interrupted and you want to resume without losing already completed runs)
        if os.path.exists(filepath):
            with open(filepath, "r",encoding="utf-8") as f:
                summary = json.load(f)
            if cfg.use_checkpoint:
                output_file = os.path.join(cfg.hydra.run.dir, 'logs',str(i),"wandb","latest-run","files","output.log")
                cfg.training.checkpoint_path = utils.extract_ckpt_from_log(output_file)
            elif "test_ood_f1_score" in summary:
                print(f"skipped run {i} as it already exists")
                continue

        logger = WandbLogger(
            project=cfg.logger.project, 
            save_dir=os.path.join(cfg.hydra.run.dir, 'logs',str(i)),
            name=f"{cfg.training.experiment_group}_{cfg.training.experiment_name}_{i}",
            id=None,
            resume='never',
            reinit=True
        )
        
        logger.experiment.config.update({"run": i})
        if isinstance(logger, WandbLogger):
            wandb.login()
            logger.experiment.config["run"] = i

        # set up callbacks
        callbacks=[ModelSummary(max_depth=2),
                                        ModelCheckpoint(monitor="val_f1_score", mode="max", save_top_k=1),
                                        LearningRateMonitor(logging_interval='epoch'),
                                        EarlyStopping(monitor="val_f1_score", mode="max",patience=30)
                                        ]
        print("Callbacks from config:", cfg.callbacks)
        callbacks += [utils.__dict__[callback_name]() 
        for callback_name in cfg.callbacks
        ]#initilze custom callbacks from utils.py based on callback list from config file
        if cfg.datamodule.name == "tolkach":
            cfg.datamodule.CHA_split_id = int(cha_splits[i])
            cfg.datamodule.CHA_split_val_id = int(cha_splits_val[i])
            cfg.datamodule.WNS_split_id = int(wns_splits[i])
            cfg.datamodule.WNS_split_val_id = int(wns_splits_val[i])
            cfg.datamodule.bio_class_distribution_id = i
        elif cfg.datamodule.name == "camelyon":
            cfg.datamodule.iteration = i
            
        datamodule = hydra.utils.instantiate(cfg.datamodule)(cfg)
        module: nn.Module  = hydra.utils.instantiate(cfg.module)(cfg)
        trainer = L.Trainer(max_epochs=cfg.training.max_epochs,
                            accelerator=cfg.training.device,
                            precision=cfg.training.precision,
                            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
                            logger=logger,
                            callbacks=callbacks,
                            limit_train_batches=cfg.training.limit_train_batches,
                            inference_mode=False #otherwise TTT setting wont work
                            #profiler="advanced"
                            )
        trainer.fit(
            module,
            datamodule=datamodule,
            ckpt_path=cfg.training.checkpoint_path if cfg.training.checkpoint_path else None
        )
        datamodule.setup(stage="test")
        results_run = {}

        # Now you can add keys freely
        trainer.model.current_test_dataset = "id"
        results_run["test_id"] = trainer.test(ckpt_path="best",verbose=True,dataloaders=datamodule.test_id_dataloader())
        
        trainer.model.current_test_dataset = "ood"
        results_run["test_ood"] = trainer.test(ckpt_path="best",verbose=True,dataloaders=datamodule.test_ood_dataloader())
        results[i] = results_run
        with open(os.path.join(cfg.hydra.run.dir, 'logs', str(i),'results_run.json'), 'w') as f:
            json.dump(results_run, f, indent=2)
        if isinstance(logger, WandbLogger):
            wandb.finish()
    with open(os.path.join(cfg.hydra.run.dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    
if __name__ == "__main__":
    main()