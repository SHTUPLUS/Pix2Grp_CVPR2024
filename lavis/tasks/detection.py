"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json

import torch


from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.evaluation.sgg_oi_eval import OpenImageSGGEvaluator
import lavis.tasks.evaluation.comm as comm 

@registry.register_task("detection")
class DetectionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, cate_dict_url=''):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

        
        with open(cate_dict_url, 'r') as f:
            self.cate_dict = json.load(f)

        self.evaluator = OpenImageSGGEvaluator(self.cate_dict, eval_post_proc=True, eval_types=['bbox'])
        self.gather_res = None 


    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            cate_dict_url=run_cfg.get("cate_dict_url", "/mnt/petrelfs/lirongjie/project/LAVIS/cache/openimages/open-imagev6/annotations/categories_dict.json")
         )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        predictions, ground_truths, image_info = model.generate(
            samples,
            use_nucleus_sampling=True,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            num_captions=1
        )

        img_ids = samples["image_id"]
        for idx, img_id in enumerate(img_ids):
            results.append({
                "predictions": predictions[idx],
                "ground_truths": ground_truths[idx],
                "image_id": img_id,
                "image_info": image_info[idx],
            })
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        predictions = {each['image_id']: each['predictions'] for each in val_result}
        ground_truths = {each['image_id']: each['ground_truths'] for each in val_result}
        self.gather_res = self.evaluator.chunk_gather(predictions, ground_truths)
        
        if comm.get_rank() == 0:
            predictions, ground_truths = self.gather_res
            eval_res = self.evaluator.evaluate(predictions, ground_truths)
            print(split_name, eval_res)

        torch.distributed.barrier()

        return {"agg_metrics": 0.0}


    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        pass


