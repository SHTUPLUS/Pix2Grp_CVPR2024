"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os

import numpy as np
import torch
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)


        it_pair_type = None
        it_pair_type_dict = None

        eval_result_all = {}
        if is_main_process():
            eval_result = self._report_metrics(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
            )

            logging.info("ALL data:")
            logging.info(eval_result)

            eval_result_all['all'] = eval_result

            if len(data_loader.dataset.it_pair_type) > 0:
                it_pair_type = data_loader.dataset.it_pair_type
                it_pair_type_dict = data_loader.dataset.it_pair_type_dict
                for each_type in it_pair_type_dict.keys():
                    selected_idx = []
                    for image_id, itp_type in enumerate(it_pair_type):
                        if each_type in itp_type:
                            selected_idx.append(image_id)

                    selected_idx = np.array(selected_idx)
                    eval_result = self._report_metrics(
                        score_i2t[selected_idx],
                        score_t2i[selected_idx],
                        {idx: each for idx, each in enumerate(selected_idx)},
                        {idx: [each] for idx, each in enumerate(selected_idx)},
                    )
                    logging.info(f"split-{each_type}:")
                    logging.info(eval_result)
                    eval_result_all[each_type] = eval_result
                
                selected_idx = []
                for image_id, itp_type in enumerate(it_pair_type):
                    if len(itp_type) == 0:
                        selected_idx.append(image_id)

                selected_idx = np.array(selected_idx)
                eval_result = self._report_metrics(
                        score_i2t[selected_idx],
                        score_t2i[selected_idx],
                        {idx: each for idx, each in enumerate(selected_idx)},
                        {idx: [each] for idx, each in enumerate(selected_idx)},
                    )
                logging.info(f"split-pure winoground:")
                logging.info(eval_result)
                eval_result_all['pure-winoground'] = eval_result

        else:
            eval_result_all = None

        return eval_result_all

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt, **kwargs):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr2 = 100.0 * len(np.where(ranks < 2)[0]) / len(ranks)
        tr3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir2 = 100.0 * len(np.where(ranks < 2)[0]) / len(ranks)
        ir3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr2 + tr3 + tr5 + tr10) / 5
        ir_mean = (ir1 + ir2 + ir3 + ir5 + ir10) / 5
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r2": tr2,
            "txt_r3": tr3,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r2": ir2,
            "img_r3": ir3,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result
