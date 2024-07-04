"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_detection import (
    DetectionEvalDataset,
    DetectionTrainDataset,
)

from lavis.datasets.datasets.oiv6_rel_detection import (
    VisualRelationDetectionEvalDataset,
    VisualRelationDetectionDataset,
)

from lavis.common.registry import registry



@registry.register_builder("coco_detection")
class COCODetBuilder(BaseDatasetBuilder):
    train_dataset_cls = DetectionTrainDataset 
    eval_dataset_cls = DetectionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_det.yaml",
    }


@registry.register_builder("ovi6_detection")
class OVI6DetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualRelationDetectionDataset # DetectionTrainDataset
    eval_dataset_cls = VisualRelationDetectionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/oiv6/defaults_rel_det.yaml",
    }


