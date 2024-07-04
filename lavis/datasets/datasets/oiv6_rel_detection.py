# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import copy
from pathlib import Path
from collections import OrderedDict
import logging

import os
import os.path
import json

from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from torch.utils.data.dataloader import default_collate


import lavis.datasets.datasets.utils.transforms_vt as T 
from lavis.datasets.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)
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


def load_categories_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_entities = info['obj'] 
    ind_to_predicates = info['rel']
    entities_to_ind = OrderedDict({
        name: i for i, name in enumerate(ind_to_entities)
    })
    info['label_to_idx'] = entities_to_ind
    predicate_to_ind = OrderedDict({
        name: i for i, name in enumerate(ind_to_predicates)
    })
    info['predicate_to_idx'] = predicate_to_ind

    return ind_to_entities, ind_to_predicates, entities_to_ind, predicate_to_ind



class VisualRelationDetectionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.vis_root = vis_root
        ann_path = ann_paths[0]
        
        ann_path_dir = "/".join(ann_path.split('/')[:-1])
        (self.ind_to_entities,
         self.ind_to_predicates,
         self.entities_to_ind,
         self.predicate_to_ind) = load_categories_info(os.path.join(ann_path_dir, 'categories_dict.json'))
        
        self.annotation_file = json.load(open(ann_paths[0], 'r'))
        
        filtered_anno = []
        logger.info(f'init rel {len(self.annotation_file)}' )
        for anno in self.annotation_file:
            if len(torch.tensor(anno["rel"])) > 0 and len(torch.tensor(anno["bbox"])) > 0:
                filtered_anno.append(anno)

        self.annotation_file = filtered_anno
        logger.info(f'filtered rel {len(self.annotation_file)}')

        # self.annotation_file = self.annotation_file[:300]
        
        self.ids = range(len(self.annotation_file))

        self.vis_processor = vis_processor # val processor for simple version
        self.text_processor = None

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
        anno = self.annotation_file[ids]
        if anno.get('image_pth') is None:
            image_pth = os.path.join(self.vis_root, anno['img_fn'] + ".jpg")
        else:
            image_pth = anno.get('image_pth')

        image = Image.open(image_pth).convert("RGB") 

        # image = Image.fromarray(np.ones((128, 128, 3), dtype=np.uint8))
        # dict_keys(['bbox', 'det_labels', 'rel', 'img_size', 'img_fn'])
        target = prepare_anno(image, anno, self.ind_to_entities, self.ind_to_predicates)

        image, target = self.vis_processor(image, target)
        label_texts = []

        rels =  target['rel_tripets'].numpy()
        ent_acc_rel_idx = set()
        for each_r in rels:
            ent_acc_rel_idx.add(each_r[0])
            ent_acc_rel_idx.add(each_r[1])

        abs2rel = {r:i for i, r in enumerate(ent_acc_rel_idx)}

        rel_rel_trp = []
        for each_r in rels:
            trps = [abs2rel[each_r[0]], abs2rel[each_r[1]], each_r[2]]

            # debug
            # trps = [abs2rel[each_r[0]], abs2rel[each_r[1]], 2]
            # debug
            rel_rel_trp.append(trps)

        
        target['rel_tripets'] = torch.from_numpy(np.array(list(rel_rel_trp), dtype=int))
        ent_acc_rel_idx = torch.from_numpy(np.array(list(ent_acc_rel_idx), dtype=int))
        target['det_labels']= target['det_labels'][ent_acc_rel_idx]

        # debug
        # target['det_labels'] = torch.ones_like(target['det_labels']) * 2
        # debug

        target["boxes"] = target["boxes"][ent_acc_rel_idx]
        target["labels"] = target['det_labels']

        for each_l in target['det_labels']:
            l_text = self.ind_to_entities[each_l.item()]
            label_texts.append(l_text)
        target['det_labels_texts'] = label_texts
        target["labels_texts"] =  label_texts
        
        predc_label_texts = []
        for each_l in target['rel_tripets']:
            predc_label_texts.append(self.ind_to_predicates[each_l[-1].item()])
        target['rel_labels_texts'] = predc_label_texts
        # logger.info([each['name'] for each in self.coco.cats.values()])
        # dict_keys('boxes': 0-1 xywh,
        # 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'])

        return {
            "instance_id": ids,
            "image_id": anno['img_fn'],
            "image_pth": image_pth,
            "image": image,
            "target": target,
        }


zero_shot_ent = [226, 395, 548, 386, 400, 25, 44, 113, 345, 93, 322, 203, 261, 499, 484, 237, 78, 589, 218, 380, 419, 569, 530, 178, 431, 23, 529, 393, 500, 324, 201, 46, 150, 190, 165, 554, 159, 278, 174, 363, 534, 121, 482, 535, 96, 428, 483, 5, 524, 85, 543, 196, 367, 596, 407, 513, 326, 331, 282, 426, 289, 172, 334, 91, 29, 369, 446, 320, 382, 290, 418, 68, 245, 319, 489, 154, 405, 131, 137, 119, 366, 556, 264, 158, 271, 38, 193, 439, 83, 84, 541, 215, 246, 141, 252, 410, 206, 256, 344, 164, 455, 36, 37, 185, 1, 436, 580, 401, 442, 127, 389, 463, 167, 45, 136, 452, 79, 28, 597, 276, 467, 450, 126, 283, 19, 101, 16, 473, 217, 236, 557, 149, 27, 301, 184, 447, 295, 342, 312, 43, 69, 14, 102, 250, 346, 272, 350, 186, 590, 98, 191, 414, 553, 116, 335, 375, 516, 420, 40, 394, 352, 90, 429, 495, 417, 303, 514, 95, 73, 94, 115, 47, 104, 408, 311, 251, 48, 327, 373, 143]
zero_shot_pred =  [8, 4, 28, 13, 22, 10, 3, 23, 7, 26, 27, 1]
class VisualRelationDetectionDataset(VisualRelationDetectionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_processor = vis_processor # val processor for simple version
        self.text_processor = None

        # self.annotation_file = filtered_anno
        if 'openvoc_ent' in ann_paths[0]:
            logger.info('entity unseen open-vocabulary setting.')
            filtered_anno = []
            for anno in self.annotation_file:
                filtered_rel = []
                for each_rel in anno["rel"]:
                    if anno["det_labels"][each_rel[0]] in zero_shot_ent or anno["det_labels"][each_rel[1]] in zero_shot_ent:
                        continue
                    filtered_rel.append(each_rel)
                
                if len(filtered_rel) > 0:
                    new_anno = copy.deepcopy(anno)
                    new_anno['rel'] = filtered_rel
                    filtered_anno.append(new_anno)

            logger.info(f'filtered rel {len(self.annotation_file)} -> {len(filtered_anno)}')
            self.annotation_file = filtered_anno
            self.ids = range(len(self.annotation_file))

        if 'zs_pred' in ann_paths[0]:
            logger.info('predicate unseen open-vocabulary setting.')
            filtered_anno = []
            for anno in self.annotation_file:
                filtered_rel = []
                for each_rel in anno["rel"]:
                    if each_rel[-1] in zero_shot_pred:
                        continue
                    filtered_rel.append(each_rel)
                
                if len(filtered_rel) > 0:
                    new_anno = copy.deepcopy(anno)
                    new_anno['rel'] = filtered_rel
                    filtered_anno.append(new_anno)

            logger.info(f'filtered rel {len(self.annotation_file)} -> {len(filtered_anno)}')
            self.annotation_file = filtered_anno
            self.ids = range(len(self.annotation_file))

        # self.annotation_file = self.annotation_file[:400]
        self.ids = range(len(self.annotation_file))




def prepare_anno(image, targets, ind_to_entities, ind_to_predicates):

    w, h = image.size
    

    image_id = targets['img_fn']
    boxes = targets["bbox"]

    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) # xyxy

    classes = targets["det_labels"]
    classes = torch.tensor(classes, dtype=torch.int64)

    rel_triplets =  targets["rel"]
    rel_triplets = torch.tensor(rel_triplets, dtype=torch.int64).reshape(-1, 3)

    target = {}
    target["boxes"] = boxes
    target['image'] = image
    target["labels"] = classes
    target["det_labels"] = classes
    target["labels_texts"] = [ind_to_entities[each.item()] for each in classes]
    target["rel_labels_texts"] = [ind_to_predicates[each[-1].item()] for each in rel_triplets]
    target["rel_tripets"] = rel_triplets
    target["image_id"] = image_id

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return target
