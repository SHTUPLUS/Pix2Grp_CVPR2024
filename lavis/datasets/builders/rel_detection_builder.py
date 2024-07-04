"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.oiv6_rel_detection import (
    VisualRelationDetectionEvalDataset,
    VisualRelationDetectionDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.vg_rel_detection import VGVRDDataset, VGVRDEvalDataset

from lavis.datasets.datasets.psg_rel_detection import (
    PSGVRDDataset,
    PSGVRDEvalDataset,
)


@registry.register_builder("oiv6_rel_detection")
class VrdOIV6DetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualRelationDetectionDataset # VisualRelationDetectionEvalDataset VisualRelationDetectionDataset 
    eval_dataset_cls = VisualRelationDetectionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/oiv6/defaults_rel_det.yaml",
    }


@registry.register_builder("oiv6_rel_detection_zs_pred")
class VrdOIV6DetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualRelationDetectionDataset # VisualRelationDetectionEvalDataset VisualRelationDetectionDataset 
    eval_dataset_cls = VisualRelationDetectionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/oiv6/defaults_rel_det_zs_pred.yaml",
    }



@registry.register_builder("gqa_rel_detection")
class VrdGQADetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualRelationDetectionDataset 
    eval_dataset_cls = VisualRelationDetectionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults_rel_det.yaml",
    }

@registry.register_builder("sgg_concate_rel_detection")
class VrdConcateDetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualRelationDetectionDataset 
    eval_dataset_cls = VisualRelationDetectionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sgg_concat/defaults_rel_det.yaml",
    }



@registry.register_builder("psg_rel_detection")
class VrdPSGDetBuilder(BaseDatasetBuilder):

    train_dataset_cls = PSGVRDDataset 
    eval_dataset_cls = PSGVRDEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/psg/defaults_rel_det.yaml",
    }




@registry.register_builder("psg_rel_detection_zs_pred")
class VrdPSGDetBuilder(BaseDatasetBuilder):

    train_dataset_cls = PSGVRDDataset 
    eval_dataset_cls = PSGVRDEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/psg/defaults_rel_det_zs_pred.yaml",
    }


@registry.register_builder("vg_rel_detection_train")
class VrdTrainVGDetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVRDDataset 
    eval_dataset_cls = VGVRDEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_rel_det_train.yaml",
    }


@registry.register_builder("vg_rel_detection_eval")
class VrdEvalVGDetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVRDDataset 
    eval_dataset_cls = VGVRDEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_rel_det_eval.yaml",
    }



@registry.register_builder("vg_rel_detection")
class VrdVGDetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVRDDataset 
    eval_dataset_cls = VGVRDEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_rel_det.yaml",
    }



@registry.register_builder("vg_rel_detection_zs_pred")
class VrdZSPredVGDetBuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVRDDataset 
    eval_dataset_cls = VGVRDEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_rel_det_train_zs_pred.yaml",
    }

