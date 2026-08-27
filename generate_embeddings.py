#!/usr/bin/env python3
import torchvision
import PIL.Image
import argparse
import utils
import os
import torch
import tqdm
import random

#python generate_embeddings.py --model hoptimus1 --outputfolder $OUTPUT_ROOT/classification_patch/tolkach/debugging/test/embs_hoptimus1 --datadir $DATA_ROOT/PathoROB/tolkach_esca
def get_args():
    parser = argparse.ArgumentParser(
        description="generating embeddings"
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
        "--model",
        type=str,
        required=True,
        help="backbone model to use"
    )

    parser.add_argument(
        "--file_suffix",
        type=str,
        default=".jpg",
        help="file suffix of images"
    )

    parser.add_argument(
        "--check_corrupted",
        type=bool,
        default=False,
        help="load and check existing .pt"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print("Data dir:", args.datadir)
    print("Output folder:", args.outputfolder)
    if not os.path.exists(args.outputfolder):
        os.makedirs(args.outputfolder)
    im_list = os.listdir(args.datadir)
    random.shuffle(im_list)
    model = utils.init_backbone(backbone_name=args.model, num_classes=1)
    transforms = model.get_transforms()
    for im in tqdm.tqdm(im_list):

        if args.check_corrupted:
            if os.path.exists(os.path.join(args.outputfolder,im.rsplit(".",1)[0]+".pt")):
                try:
                    torch.load(os.path.join(args.outputfolder,im.rsplit(".",1)[0]+".pt"))
                    continue
                except:
                    print(f'regenerating corrupted file: {os.path.join(args.outputfolder,im.rsplit(".",1)[0]+".pt")}')
        else:
            if os.path.exists(os.path.join(args.outputfolder,im.rsplit(".",1)[0]+".pt")):
                continue

        img = PIL.Image.open(os.path.join(args.datadir,im))
        img = transforms(img)
        emb = model.extract_tokens(img.unsqueeze(0))
        torch.save(emb,os.path.join(args.outputfolder,im.rsplit(".",1)[0]+".pt"))