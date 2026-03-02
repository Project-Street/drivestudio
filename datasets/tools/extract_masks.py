"""
@file   extract_masks.py
@brief  Extract semantic mask using SegFormer via HuggingFace Transformers.

Using SegFormer B5 fine-tuned on Cityscapes (83.2% mIoU).
Model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024

Installation:
    pip install transformers

    The model is automatically downloaded from HuggingFace on first run.

Usage:
    python datasets/tools/extract_masks.py \
        --data_root data/waymo/processed/training \
        --split_file data/waymo_example_scenes.txt \
        --process_dynamic_mask
"""

import os
import numpy as np
import imageio
import torch
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Cityscapes 19-class index mapping
# sky: 10, person: 11, rider: 12, car: 13, truck: 14, bus: 15, motorcycle: 17, bicycle: 18
VEHICLE_CLASSES = [13, 14, 15]         # car, truck, bus
HUMAN_CLASSES = [11, 12, 17, 18]       # person, rider, motorcycle, bicycle
SKY_CLASS = 10


def inference(model, image_processor, fpath, device):
    image = imageio.imread(fpath).astype(np.float32) / 255.0  # [H, W, 3], [0, 1]
    h, w = image.shape[:2]

    inputs = image_processor(images=image, return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # [1, 19, H', W']
    logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]
    return mask


if __name__ == "__main__":
    parser = ArgumentParser()
    # Data configs
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space.",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        '--process_dynamic_mask',
        action='store_true',
        help="Whether to process fine dynamic masks in addition to sky masks",
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")

    # Model configs
    parser.add_argument(
        '--model_name',
        type=str,
        default='nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
        help='HuggingFace model name for SegFormer',
    )
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        split_file = open(args.split_file, "r").readlines()[1:]
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in split_file]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    image_processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    for scene_i, scene_id in enumerate(tqdm(scene_ids_list, desc='Extracting Masks')):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)

        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        os.makedirs(sky_mask_dir, exist_ok=True)

        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")
            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            os.makedirs(all_mask_dir, exist_ok=True)
            os.makedirs(human_mask_dir, exist_ok=True)
            os.makedirs(vehicle_mask_dir, exist_ok=True)

        flist = sorted(glob(os.path.join(img_dir, '*')))
        for fpath in tqdm(flist, desc=f'scene[{scene_id}]', disable=not args.verbose):
            fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]

            if args.ignore_existing and os.path.exists(os.path.join(args.data_root, scene_id, "fine_dynamic_masks")):
                continue

            mask = inference(model, image_processor, fpath, args.device)

            # Save sky mask
            sky_mask = np.isin(mask, [SKY_CLASS])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

            if args.process_dynamic_mask:
                # Save human mask (refined by coarse dynamic mask)
                rough_human_mask = (imageio.imread(os.path.join(rough_human_mask_dir, f"{fbase}.png")) > 0)
                human_mask = np.isin(mask, HUMAN_CLASSES)
                valid_human_mask = np.logical_and(human_mask, rough_human_mask)
                imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"), valid_human_mask.astype(np.uint8) * 255)

                # Save vehicle mask (refined by coarse dynamic mask)
                rough_vehicle_mask = (imageio.imread(os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")) > 0)
                vehicle_mask = np.isin(mask, VEHICLE_CLASSES)
                valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                imageio.imwrite(os.path.join(vehicle_mask_dir, f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8) * 255)

                # Save combined dynamic mask
                valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all_mask.astype(np.uint8) * 255)
