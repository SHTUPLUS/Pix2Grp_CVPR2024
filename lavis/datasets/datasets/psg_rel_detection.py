# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import copy
import logging
from pathlib import Path
from collections import OrderedDict

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


class PSGVRDEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.vis_root = vis_root
        ann_path = ann_paths[0]
        ann_path_dir = "/".join(ann_path.split('/')[:-1])

        logger.info(f"build dataset {ann_path}")
        
        (self.ind_to_entities,
         self.ind_to_predicates,
         self.entities_to_ind,
         self.predicate_to_ind) = load_categories_info(os.path.join(ann_path_dir, 'categories_dict.json'))

        self.annotation_file = json.load(open(ann_paths[0], 'r'))
        test_image_ids = self.annotation_file['test_image_ids']

        self.annotation_file = self.annotation_file['data']

        filtered = []
        for idx, each in enumerate(self.annotation_file):
            if len(each['relations']) > 0:
                filtered.append(each)

        print("filtering no relation data:", len(self.annotation_file), "->", len(filtered))
        self.annotation_file = filtered
        
        # ##########################
        # only keep zero-shot entity
        # filtered = []
        # for idx, each in enumerate(self.annotation_file):
        #     new_rel = []
        #     for rel in each['relations']:
        #         if rel[-1] in PSGVRDDataset.zeroshot_predicate:
        #             new_rel.append(rel)
        #     if len(new_rel) > 0:
        #         new_anno = copy.deepcopy(each)
        #         new_anno['relations'] = new_rel
        #         filtered.append(new_anno)
        
        # print(len(self.annotation_file), len(filtered))
        # print("filtering unsee class rel data:", len(self.annotation_file), "->", len(filtered))
        # self.annotation_file = filtered

        # ##########################
        # self.annotation_file = self.annotation_file[:256]

        ##########################


        self.ids = range(len(self.annotation_file))

        self.vis_processor = vis_processor  # val processor for simple version
        self.text_processor = None

        self.remove_no_rel_ent = False


    def _add_instance_ids(self, key="instance_id"):
        pass

    def collater(self, samples):
        img_tensor = default_collate([each['image'] for each in samples])
        anno = [each['target'] for each in samples]
        instance_id = [each['instance_id'] for each in samples]
        image_pth = [each['image_pth'] for each in samples]
        image_id = [each['image_id'] for each in samples]
        return {'image': img_tensor, 'targets': anno, "image_id": image_id,
                "instance_id": instance_id, "image_pth": image_pth}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        ids = self.ids[index]
        anno = self.annotation_file[ids]
        image_pth = os.path.join(self.vis_root, anno['file_name'])
        image = Image.open(image_pth).convert("RGB")

        # image = Image.fromarray(np.ones((128, 128, 3), dtype=np.uint8))
        # dict_keys(['bbox', 'det_labels', 'rel', 'img_size', 'img_fn'])
        target = prepare_anno(
            image, anno, self.ind_to_entities, self.ind_to_predicates)

        image, target = self.vis_processor(image, target)
        label_texts = []
        
        if self.remove_no_rel_ent:
            rels = target['rel_tripets'].numpy()
            ent_acc_rel_idx = set()
            for each_r in rels:
                ent_acc_rel_idx.add(each_r[0])
                ent_acc_rel_idx.add(each_r[1])
            abs2rel = {r: i for i, r in enumerate(ent_acc_rel_idx)}
            rel_rel_trp = []
            for each_r in rels:
                trps = [abs2rel[each_r[0]], abs2rel[each_r[1]], each_r[2]]
                rel_rel_trp.append(trps)

            target['rel_tripets'] = torch.from_numpy(
                np.array(list(rel_rel_trp), dtype=int))
            ent_acc_rel_idx = torch.from_numpy(
                np.array(list(ent_acc_rel_idx), dtype=int))
            target['det_labels'] = target['det_labels'][ent_acc_rel_idx]
            target["boxes"] = target["boxes"][ent_acc_rel_idx]
            target["labels"] = target['det_labels']

        for each_l in target['det_labels']:
            l_text = self.ind_to_entities[each_l.item()]
            label_texts.append(l_text)
        target['det_labels_texts'] = label_texts
        target["labels_texts"] = label_texts

        predc_label_texts = []
        for each_l in target['rel_tripets']:
            predc_label_texts.append(self.ind_to_predicates[each_l[-1].item()])
        target['rel_labels_texts'] = predc_label_texts
        # print([each['name'] for each in self.coco.cats.values()])
        # dict_keys('boxes': 0-1 xywh,
        # 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'])

        return {
            "instance_id": ids,
            "image_id": anno['image_id'],
            "image_pth": image_pth,
            "image": image,
            "target": target,
        }


class PSGVRDDataset(PSGVRDEvalDataset):
    zeroshot_predicate = [16, 23, 2, 14, 15, 26, 46, 4, 22, 1, 48, 37, 5, 11, 47, 12]
    # [16, 24, 2, 36, 14, 15, 6, 5, 46, 4, 53, 22, 8, 48, 37, 51, 5, 1, 33, 47, 12, 25, 0, 19, 29]

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_processor = vis_processor  # val processor for simple version
        self.text_processor = None

        if 'zs_pred' in ann_paths[0]:
            filtered_anno = []
            for idx, anno in enumerate(self.annotation_file):
                filtered_rel = []
                for rel in anno['relations']:
                    if rel[-1] in PSGVRDDataset.zeroshot_predicate:
                        continue 
                    filtered_rel.append(rel)
                
                if len(filtered_rel) > 0:
                    new_anno = copy.deepcopy(anno)
                    new_anno['relations'] = filtered_rel
                    filtered_anno.append(new_anno)

            logger.info(f'predicate open vocabulary: filtered rel {len(self.annotation_file)} -> {len(filtered_anno)}')
            self.annotation_file = filtered_anno
            self.ids = range(len(self.annotation_file))
            
        self.remove_no_rel_ent = False

def prepare_anno(image, targets, ind_to_entities, ind_to_predicates):

    w, h = image.size

    image_id = targets['image_id']

    boxes = [each['bbox'] for each in targets["annotations"]]

    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # xyxy

    classes = [each['category_id'] for each in targets["annotations"]]
    classes = torch.tensor(classes, dtype=torch.int64)

    rel_triplets = targets["relations"]
    rel_triplets = torch.tensor(rel_triplets, dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target['image'] = image
    target["labels"] = classes
    target["det_labels"] = classes
    target["labels_texts"] = [ind_to_entities[each.item()] for each in classes]
    target["rel_labels_texts"] = [ind_to_predicates[each[-1].item()]
                                  for each in rel_triplets]
    target["rel_tripets"] = rel_triplets
    target["image_id"] = image_id

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return target
