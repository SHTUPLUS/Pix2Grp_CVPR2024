"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from pprint import pprint

from PIL import Image
import msgspec

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class GQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:

            with open(ann_path, "rb") as f:
                anno_dict = msgspec.json.decode(f.read())

            for k, v in anno_dict.items():
                v['init_instance_id'] = k
                self.annotation.append(v)

            # pprint(anno_dict[k])

        # self.annotation = self.annotation[:100] # for debug
        
        print("train_set keys:", self.annotation[0].keys())
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.type2id = {"obj": 0, "attr": 1, "rel": 2, "global": 3, "cat": 4}

        for idx, ann in enumerate(self.annotation):
            ann["question_id"] = idx # question_id需要是int

        self._add_instance_ids()

        print("data_length", len(self))


    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["imageId"] + '.jpg')
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        weights = [1]

        if 'fullAnswer' in ann:
            full_answer = ann["fullAnswer"]
        else:
            full_answer = None

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "image_fname_id" : ann["imageId"],
            "question_type": ann['semantic'],
            "question_type_id": self.type2id[ann['semantic']],
            "answers": answers,
            'direct_answers': answers, 
            'full_answer': [full_answer], 
            "weights": weights,
        }

    # def _add_instance_ids(self, key="instance_id"):
    #     for idx, anno_key in enumerate(self.annotation.keys()):
    #         self.annotation[anno_key][key] = str(idx)
    #         # ann[key] = str(idx)


class GQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:

            with open(ann_path, "rb") as f:
                anno_dict = msgspec.json.decode(f.read())

            for k, v in anno_dict.items():
                v['init_instance_id'] = k
                self.annotation.append(v)

        print("val_set keys:", self.annotation[0].keys())

        # self.annotation = self.annotation[:100] # for debug
        
        # dict_keys(['semantic', 'entailed', 'equivalent', 'question', 'imageId', 'isBalanced', 'groups', 'answer', 'semanticStr', 'annotations', 'types', 'fullAnswer', 'init_instance_id'])

        for idx, ann in enumerate(self.annotation):
            ann["question_id"] = idx # question_id需要是int

        self._add_instance_ids()

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.type2id = {"obj": 0, "attr": 1, "rel": 2, "global": 3, "cat": 4}

        print("data_length", len(self))


    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["imageId"] + '.jpg')
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        if "answer" in ann:
            # answer is a string
            answer = ann["answer"]
        else:
            answer = None

        if 'fullAnswer' in ann:
            full_answer = ann["fullAnswer"]
        else:
            full_answer = None
            

        return {
            "image": image,
            "text_input": question,
            "answer": answer,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "image_fname_id" : ann["imageId"],
            "question_type": ann['semantic'],
            "question_type_id": self.type2id[ann['semantic']],
            'direct_answers': answer, 
            "fullAnswer":full_answer,
            "weights": answer,
        }
