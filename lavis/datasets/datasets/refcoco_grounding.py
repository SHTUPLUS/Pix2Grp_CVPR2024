# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from collections import OrderedDict

import os
import os.path

from PIL import Image
from pycocotools.coco import COCO
import torch
import torch.utils.data
import torchvision
import numpy as np
from pycocotools import mask as coco_mask
from torch.utils.data.dataloader import default_collate


import lavis.datasets.datasets.utils.transforms_vt as T 
from lavis.datasets.datasets.base_dataset import BaseDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "targets": ann["targets"],
                "image": sample["image"],
            }
        )


class DetectionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, image_size=384):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.vis_root = vis_root
        self.raw_anno_info = torch.load(ann_paths)
        self.anno_list = []

        for idx, anno in enumerate(self.raw_anno_info):
            img_file, _, bbox, phrase, attri = anno

            train_data = {}
            bbox = np.array(bbox, dtype=np.float32)  # xywh
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]  # xyxy

            train_data['gt_bboxes'] = bbox[None, :]
            train_data['bbox'] = bbox
            train_data['text'] = phrase

            img_id = img_file.split("/")[-1].strip(".jpg").split("_")[-1]
            train_data['image_id'] = img_id
            train_data['instance_id'] = idx

            train_data['img_path'] = os.path.join(
                self.data_prefix['img'], img_file)

            self.anno_list.append(train_data)

        self.ids = range(len(self.anno_list))
        self.image_size = image_size
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        pass

    def collater(self, samples):
        img_tensor = default_collate([each['image'] for each in samples])
        anno = [each['target'] for each in samples]
        instance_id = [each['instance_id'] for each in samples]
        image_pth = [each['image_pth'] for each in samples ]
        image_id = [each['image_id'] for each in samples ]
        return {'image': img_tensor, 'targets': anno, "image_id": image_id,
                "instance_id": instance_id, "image_pth": image_pth}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ids = self.ids[index]
        path = self.coco.loadImgs(ids)[0]["coco_url"]
        path = '/'.join(path.split('/')[-2:])

        image = Image.open(os.path.join(self.vis_root, path)).convert("RGB") 
        # dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        anno = self.coco.loadAnns(self.coco.getAnnIds(ids))

        anno = prepare_anno(image, anno)
        image, target = self.vis_processor(image, anno)

        # blip norm setting
        # mean = torch.Tensor((0.48145466, 0.4578275, 0.40821073))
        # std = torch.Tensor((0.26862954, 0.26130258, 0.27577711))
        # image_init = image * std + mean

        label_texts = []
        for each_l in target['labels']:
            l_text = self.coco.cats[each_l.item()]['name']
            label_texts.append(l_text)
        target['labels_texts'] = label_texts

        # print([each['name'] for each in self.coco.cats.values()])
        # dict_keys('boxes': 0-1 xywh,
        # 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'])

        return {
            "instance_id": ids,
            "image_id": path.split('/')[-1],
            "image_pth": os.path.join(self.vis_root, path),
            "image": image,
            # "image_init": image_init,
            "target": target,
        }

class DetectionTrainDataset(DetectionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, image_size=384):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.vis_root = vis_root
        self.coco = COCO(ann_paths[0])
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.image_size = image_size
        self.vis_processor = vis_processor
        self.text_processor = None

def make_coco_transforms(image_set, image_size, large_scale_jitter=False):
    
    # blip norm setting
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if large_scale_jitter:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.LargeScaleJitter(output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0),
                T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomSelect(
                #     T.RandomResize(scales, max_size=1333),
                #     T.Compose([
                #         T.RandomSizeCrop(384, 600),
                #         T.RandomResize([(384, 384)]),
                #     ])
                # ),
                T.RandomResize([(image_size, image_size)]),
                normalize,
            ])

    if image_set == 'val':
        if large_scale_jitter:
            return T.Compose([
                T.LargeScaleJitter(output_size=1333, aug_scale_min=1.0, aug_scale_max=1.0),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResize([(image_size, image_size)]),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')




def prepare_anno(image, targets):
    w, h = image.size
    
    image_id = [obj['image_id'] for obj in targets ]

    boxes = targets["bbox"]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2] # xywh to xyxy
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["det_labels"] = classes

    target["image_id"] = image_id

    # for conversion to coco api
    area = torch.tensor([obj["area"] for obj in targets])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in targets])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return target

