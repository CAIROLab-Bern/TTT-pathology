#!/usr/bin/env python3
"""
Pregenerate H-optimus-1 patch embeddings for the Tolkach-ESCA dataset.

Extends the plain embedding extraction in generate_embeddings.py with the
multi-magnification simulation used by AuxMagTTT (see paper Fig. 2B): the
same 20x-scanned patch is re-cropped/rescaled to imitate how it would look if
it had been scanned at 18x or 22x, before being resized back to the model's
native input size. A wider center-crop (more tissue context per pixel)
simulates the lower-power 18x view; a tighter center-crop simulates the
higher-power 22x view.

Usage (after `source .env` and setting TTT_DATA_ROOT):
    python3 generate_tolkach_embeddings.py \
        --datadir $TTT_DATA_ROOT/PathoROB/tolkach_esca \
        --outputroot $TTT_DATA_ROOT/PathoROB/precomputed_embeddings_tolkach \
        --model hoptimus1
"""
import argparse
import os
import random

import PIL.Image
import torch
import torchvision
import tqdm

import utils  # relative import

# crop size (before resizing back to the model's input resolution) used to
# simulate each magnification level from a native 20x-scanned patch.
MAGNIFICATION_CROPS = {
    "18": 248,  # wider crop -> more tissue per pixel -> lower effective magnification
    "20": 224,  # native crop, no magnification change
    "22": 200,  # tighter crop -> less tissue per pixel -> higher effective magnification
}
MAGNIFICATION_SUFFIXES = {"18": "_x18", "20": "", "22": "_x22"}


def get_args():
    parser = argparse.ArgumentParser(
        description="Pregenerate (multi-magnification) embeddings for Tolkach-ESCA"
    )
    parser.add_argument("--datadir", type=str, required=True,
                         help="Path to the flat Tolkach-ESCA .jpg patch directory")
    parser.add_argument("--outputroot", type=str, required=True,
                         help="Path under which embs_<model>[_x18/_x22] folders are written")
    parser.add_argument("--model", type=str, required=True,
                         help="backbone model to use, e.g. hoptimus1")
    parser.add_argument("--magnifications", type=str, default="18,20,22",
                         help="comma-separated subset of {18,20,22} to generate")
    parser.add_argument("--file_suffix", type=str, default=".jpg",
                         help="file suffix of input images")
    return parser.parse_args()


def build_transform(model, crop_size, resize_size):
    """Same normalization stats as model.get_transforms(), but with a
    configurable center-crop so a single native-resolution image can be
    re-used to simulate every magnification level."""
    native = model.get_transforms()
    normalize = native.transforms[-1]
    return torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.Resize(resize_size),
        torchvision.transforms.ToTensor(),
        normalize,
    ])


if __name__ == "__main__":
    args = get_args()
    magnifications = [m.strip() for m in args.magnifications.split(",")]
    for m in magnifications:
        if m not in MAGNIFICATION_CROPS:
            raise ValueError(f"Unsupported magnification '{m}', expected one of {list(MAGNIFICATION_CROPS)}")

    print("Data dir:", args.datadir)
    print("Output root:", args.outputroot)
    print("Magnifications:", magnifications)

    model = utils.init_backbone(backbone_name=args.model, num_classes=1)
    native_resize = model.get_transforms().transforms[1].size  # Resize target used at 20x

    im_list = os.listdir(args.datadir)
    random.shuffle(im_list)

    for m in magnifications:
        outputfolder = os.path.join(args.outputroot, f"embs_{args.model}{MAGNIFICATION_SUFFIXES[m]}")
        os.makedirs(outputfolder, exist_ok=True)
        transform = build_transform(model, MAGNIFICATION_CROPS[m], native_resize)

        print(f"\n=== magnification {m}x -> {outputfolder} ===")
        for im in tqdm.tqdm(im_list):
            out_path = os.path.join(outputfolder, im.rsplit(".", 1)[0] + ".pt")
            if os.path.exists(out_path):
                continue
            img = PIL.Image.open(os.path.join(args.datadir, im))
            img = transform(img)
            emb = model.extract_tokens(img.unsqueeze(0))
            torch.save(emb, out_path)
