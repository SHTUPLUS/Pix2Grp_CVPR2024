"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import torch


from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.detection import DetectionTask
from lavis.tasks.evaluation.sgg_oi_eval import OpenImageSGGEvaluator
import lavis.tasks.evaluation.comm as comm 

logger = logging.getLogger(__name__)
@registry.register_task("relation_detection")
class RelationDetectionTask(DetectionTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, 
                 report_metric=True, experiments_mode='sggen',
                 generation_mode='sampling', cate_dict_url="", 
                 zeroshot_cfg=None):
        super().__init__(num_beams, max_len, min_len, evaluate, report_metric, cate_dict_url)

        logger.info(f"max_len {max_len}, min_len {min_len}")
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.generation_mode = generation_mode
        self.use_nucleus_sampling = True
        if generation_mode == 'sampling':
            self.use_nucleus_sampling = True
        elif generation_mode == 'search':
            self.use_nucleus_sampling = False
        print("use_nucleus_sampling", self.use_nucleus_sampling)
        self.report_metric = report_metric


        self.evaluator = OpenImageSGGEvaluator(self.cate_dict, 
                                               eval_post_proc=False, 
                                               eval_types=['bbox', 'relation'],
                                               zeroshot_cfg=zeroshot_cfg,)
        self.gather_res = None 
        self.experiments_mode = experiments_mode


    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        
        if self.experiments_mode == 'sggen':
            predictions, ground_truths, image_info = model.generate(
                samples,
                use_nucleus_sampling=self.use_nucleus_sampling,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                repetition_penalty=1.5,
                num_captions=1
            )
        elif self.experiments_mode == 'sgcls':
            predictions, ground_truths, image_info = model.sgg_cls(
                samples,
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

        eval_res = {"agg_metrics": 0.0}
        if comm.get_rank() == 0:
            predictions, ground_truths = self.gather_res
            eval_res_all = self.evaluator.evaluate(predictions, ground_truths)
            print(split_name, eval_res)
            eval_res.update(eval_res_all)

        torch.distributed.barrier()

        return eval_res

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        pass
    
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        
        experiments_mode = run_cfg.get("experiments_mode", 'sggen')
        generation_mode = run_cfg.get("generation_mode", 'sampling')
        cate_dict_url =  run_cfg.get("cate_dict_url", '')
        zeroshot_cfg = run_cfg.get("zeroshot_cfg", None)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            experiments_mode=experiments_mode,
            generation_mode=generation_mode,
            cate_dict_url=cate_dict_url,
            zeroshot_cfg=zeroshot_cfg
        )
