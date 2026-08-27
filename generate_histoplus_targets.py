#!/usr/bin/env python3
import torchvision
import PIL.Image
import argparse
import utils
import os
import torch
import tqdm
from histoplus.helpers.segmentor import CellViTSegmentor

def get_args():
    parser = argparse.ArgumentParser(
        description="generating histoplus targets"
    )

    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Path to input dataset directory"
    )

    parser.add_argument(
        "--outputfolder",
        type=str,
        required=True,
        help="Path to output / checkpoints directory"
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="cuda, mps, or cpu"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    hp = CellViTSegmentor.from_histoplus(
            mpp=0.5,
            mixed_precision=True,
            inference_image_size=224,
            )
    hp.model.eval()
    if not os.path.exists(args.outputfolder):
        os.makedirs(args.outputfolder)
    print("Data dir:", args.datadir)
    print("Output folder:", args.outputfolder)
    transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    im_list = os.listdir(args.datadir)

    for im in tqdm.tqdm(im_list):
        if os.path.exists(os.path.join(args.outputfolder,im.split(".jpg")[0]+".pt")):
            continue
        img = PIL.Image.open(os.path.join(args.datadir,im))
        img = transforms(img)
        emb = hp.model(img.half().unsqueeze(0).to(args.device)) #get histoplus target and move to cpu
        
        torch.save(emb,os.path.join(args.outputfolder,im.split(".jpg")[0]+".pt"))