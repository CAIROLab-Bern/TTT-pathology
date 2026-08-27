#!/usr/bin/env python3
"""
Flatten a freshly-extracted Tolkach-ESCA download into the single directory of
uniquely-named patches expected by configs/datamodule/tolkach_ubelix_precomputed*.yaml.

The raw Zenodo archives (https://doi.org/10.5281/zenodo.7548828) extract into
    <medical_center>/<BIO_CLASS>/<bio_class_lower>.<n>.jpg
e.g. VALSET1_UKK/SH_OES/sh_oes.168.jpg -- the case/slide a patch belongs to is not
encoded in that path. metadata/tolkach_esca_metadata.csv (bundled in this repo) maps
each (medical_center, bio_class, patch_id) triple to its case ("slide_id") and to the
final flat filename ("im_uuid") used throughout the rest of the codebase, so this
script looks each raw file up in the metadata and copies it under its im_uuid name.

Usage:
    python3 prepare_tolkach_esca.py \
        --rawdir $TTT_DATA_ROOT/PathoROB/tolkach_esca_raw \
        --outputdir $TTT_DATA_ROOT/PathoROB/tolkach_esca
"""
import argparse
import os
import shutil

import pandas as pd
import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Flatten extracted Tolkach-ESCA archives into the expected layout"
    )
    parser.add_argument("--rawdir", type=str, required=True,
                         help="Directory containing the extracted VALSET*/<CLASS>/*.jpg tree")
    parser.add_argument("--outputdir", type=str, required=True,
                         help="Flat output directory (e.g. $TTT_DATA_ROOT/PathoROB/tolkach_esca)")
    parser.add_argument("--metadata", type=str, default="metadata/tolkach_esca_metadata.csv",
                         help="Path to tolkach_esca_metadata.csv")
    parser.add_argument("--move", action="store_true",
                         help="Move instead of copy (frees up space, but --rawdir is consumed)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outputdir, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    lookup = {
        (row.medical_center, row.bio_class, row.patch_id): row.im_uuid
        for row in metadata.itertuples()
    }

    transfer = shutil.move if args.move else shutil.copy2
    found, missing = 0, 0
    for medical_center in sorted(os.listdir(args.rawdir)):
        center_dir = os.path.join(args.rawdir, medical_center)
        if not os.path.isdir(center_dir):
            continue
        for bio_class in sorted(os.listdir(center_dir)):
            class_dir = os.path.join(center_dir, bio_class)
            if not os.path.isdir(class_dir):
                continue
            files = os.listdir(class_dir)
            for fname in tqdm.tqdm(files, desc=f"{medical_center}/{bio_class}"):
                patch_id = fname.rsplit(".", 1)[0]  # e.g. "sh_oes.168"
                key = (medical_center, bio_class, patch_id)
                if key not in lookup:
                    missing += 1
                    continue
                out_path = os.path.join(args.outputdir, lookup[key] + ".jpg")
                if not os.path.exists(out_path):
                    transfer(os.path.join(class_dir, fname), out_path)
                found += 1

    print(f"Placed {found} patches into {args.outputdir} ({missing} raw files had no metadata match)")
