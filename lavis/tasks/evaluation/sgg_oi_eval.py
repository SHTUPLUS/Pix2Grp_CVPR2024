import contextlib
import copy
import io
import logging
from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from lavis.tasks.evaluation.vg_sgg_eval_tools import bbox_overlaps, argsort_desc, intersect_2d
from lavis.tasks.evaluation.logger import create_table_with_header, create_small_table
from .sgg_vg_eval import classic_vg_sgg_evaluation
import lavis.tasks.evaluation.comm as comm

logger = logging.getLogger(__name__)

class OpenImageSGGEvaluator():
    """
    inherits from the VG evaluator since most of preprocessing
    are shared with the initial SGG
    """

    def __init__(self, meta, eval_post_proc=False, eval_types=['bbox', 'relation'], zeroshot_cfg=None):

        super(OpenImageSGGEvaluator, self).__init__()
        self._cpu_device = torch.device("cpu")
        self.meta = meta
        self.eval_post_proc = eval_post_proc
        self.logger = logging.getLogger("lavis." + __name__)

        self._distributed = True if comm.get_world_size() > 1 else False
        self.eval_types = eval_types
        self.zeroshot_cfg = zeroshot_cfg

    def chunk_gather(self, predictions, groundtruths):
        """
        # gather things in trucks
        # due to the relation prediction is larger than detection (we may take 2048 relation pairs)
        # to avoid the OOM if GPU, we do the gather in trucks
        Args:
            predictions (dict): img_id: predict
            groundtruths (_type_): img_id: gt

        Returns:
            _type_: _description_
        """

        print(comm.get_rank(), ": before gather", len(predictions))

        # print("before gather", len(groundtruths))


        if self._distributed:
            torch.cuda.empty_cache()
            predictions_gathered = OrderedDict()    
            groundtruths_gathered = OrderedDict()

            assert len(groundtruths) == len(predictions)

            chunk_size = 10

            id_list = [ids for ids in groundtruths.keys()]

            for truck_num in range(len(id_list) // chunk_size + 1):

                gts = comm.gather({
                    sel_id: groundtruths[sel_id]
                    for sel_id in id_list[truck_num * chunk_size: (truck_num + 1) * chunk_size]
                }, dst=0)

                preds = comm.gather({
                    sel_id: predictions[sel_id]
                    for sel_id in id_list[truck_num * chunk_size: (truck_num + 1) * chunk_size]
                }, dst=0)

                for sel_id in id_list[truck_num * chunk_size: (truck_num + 1) * chunk_size]:
                    predictions.pop(sel_id)
                    groundtruths.pop(sel_id)
                
                torch.cuda.empty_cache()

                def move_to_cpu(data):
                    for i in range(len(data)):
                        for k in data[i].keys():
                            data[i][k] = data[i][k].to(torch.device('cpu'))
                    return data
            
                if comm.is_main_process():
                    for p, gt in zip(move_to_cpu(preds), move_to_cpu(gts)):
                        predictions_gathered.update(p)
                        groundtruths_gathered.update(gt)
                    # print(comm.get_rank(), ": update main proc")
                

            print("chunk gather done")
            # only main process do evaluation
            if comm.get_rank() != 0:
                print("chunk gather return")
                return None, None
            else:
                predictions_list = []
                groundtruths_list = []
                for im_id, pred in predictions_gathered.items():
                    predictions_list.append(pred.to(torch.device('cpu')))
                    groundtruths_list.append(groundtruths_gathered[im_id].to(torch.device('cpu')))

                print("after gather", len(predictions_list))
                return predictions_list, groundtruths_list
        else:
            # no distributed mode
            predictions_list = []
            groundtruths_list = []
            for im_id, preds in predictions.items():
                predictions_list.append(preds.to(torch.device('cpu')))
                groundtruths_list.append(groundtruths[im_id].to(torch.device('cpu')))

            return predictions_list, groundtruths_list

    def evaluate(self, predictions, groundtruths):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """

        eval_types = self.eval_types

        entities_categories = self.meta['obj']
        predicates_categories = self.meta['rel']
        self._results = dict()
        # evaluate the entities_box

        if "bbox" in eval_types:
            # create a Coco-like object that we can use to evaluate detection!
            
            self._results["bbox"] = self.det_coco_eval(predictions, groundtruths, entities_categories)
            
            logger.info("\n" + "=="*23)
            logger.info("det proposal eval:")
            self._results["bbox_proposal"] = self.det_coco_eval(predictions, groundtruths, entities_categories, proposal_eval=True)

        if 'relation' in eval_types:
            zeroshot_predicate = None
            zs_triplets_dir = None
            if  self.zeroshot_cfg is not None:
                zeroshot_predicate = self.zeroshot_cfg.zs_predicate
                zs_triplets_dir = self.zeroshot_cfg.zs_triplets
                
            # "/mnt/petrelfs/lirongjie/project/LAVIS/cache/openimages/open-imagev6/annotations/zeroshot_triplet.pytorch"
            # [8, 4, 28, 20, 9, 13, 18, 22, 10, 3, 23, 29, 26, 5, 24, 11, 21, 15, 14, 27, 25]
            # vg_zs_triplets_dir ='/mnt/petrelfs/lirongjie/project/LAVIS/cache/vg/vg_sgg_anno/zeroshot_triplet_lavis.pytorch'
            # vg_zeroshot_predicate = [3, 5, 25, 35, 48, 0, 23, 20, 31, 42, 34, 11, 4, 40, 1, 41, 18, 44, 36, 12, 8, 2, 16, 27, 22, 10, 26, 6, 9, 14]
            # zs_triplets_dir = vg_zs_triplets_dir

            _, vg_sgg_eval_res = classic_vg_sgg_evaluation(predictions,
                                                           groundtruths,
                                                           entities_categories,
                                                           predicates_categories,
                                                           zs_triplets_dir,
                                                           zeroshot_predicate)

            self._results['sgg_vg_metrics'] = vg_sgg_eval_res

            # transform the initial prediction into oi predition format
            packed_results = adapt_results(groundtruths, predictions)

            result_str_tmp, \
                result_dict = oi_sgg_evaluation(
                    packed_results, predicates_categories, self.eval_post_proc
                )

            self._results['sgg_oi_metrics'] = result_dict
            logger.info(result_str_tmp)

        return copy.deepcopy(self._results)

    def det_coco_eval(self, predictions, groundtruths, entities_categories, proposal_eval=False):
        """
        the wrap for coco api detection evaluations

        Args:
            predictions (_type_): _description_
            groundtruths (_type_): _description_
            entities_categories (_type_): _description_
            anns (_type_): _description_
            proposal_eval (bool, optional): set true for evualation the box only, 
            the category information will be removed. Defaults to False.
        """
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist()  # integer
            boxes = gt.bbox.tolist()  # xyxy
            for cls_id, box in zip(labels, boxes):
                if proposal_eval:
                    cls_id = 0

                anns.append({
                        'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                        # xywh
                        'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                        'category_id': cls_id,
                        'id': len(anns),
                        'image_id': image_id,
                        'iscrowd': 0,
                    })
        fauxcoco = COCO()

        if proposal_eval:
            entities_categories = ['a', 'b']
            
        fauxcoco.dataset = {
                'info': {'description': 'use coco script for vg detection evaluation'},
                'images': [{'id': i} for i in range(len(groundtruths))],
                'categories': [
                    {'supercategory': 'person', 'id': i, 'name': name}
                    for i, name in enumerate(entities_categories) if name != '__background__'
                ],
                'annotations': anns,
            }
        fauxcoco.createIndex()

            # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert(
                    'xywh').bbox.detach().cpu().numpy()  # xywh
            score = prediction.get_field(
                    'pred_scores').detach().cpu().numpy()  # (#objs,)
            label = prediction.get_field(
                    'pred_labels').detach().cpu().numpy()  # (#objs,)
            image_id = np.asarray([image_id] * len(box))

            if proposal_eval:
                label = np.zeros_like(label)

            cocolike_predictions.append(
                    np.column_stack((image_id, box, score, label))
                )
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            # evaluate via coco API
            res = fauxcoco.loadRes(cocolike_predictions)
            coco_eval = COCOeval(fauxcoco, res, 'bbox')
            coco_eval.params.imgIds = list(range(len(groundtruths)))
            coco_eval.evaluate()
            coco_eval.accumulate()
            # obtain the coco api printed output to the string
            coco_eval.summarize()
        
        coco_summary = redirect_string.getvalue()
        logger.info(f"\n{coco_summary}")

        per_class_eval_recall = []
        for cate_id in coco_eval.params.catIds:
            per_class_eval_recall.append(coco_eval.eval['recall'][0][cate_id][0][-1])

        per_class_eval_recall_str = " ".join([ f"({entities_categories[cate_id]}:{cate_perf:.3f})" 
                                            for cate_id, cate_perf in enumerate(per_class_eval_recall)])
        logger.info(f"per class detection recall:\n{per_class_eval_recall_str}")

        res = self._derive_coco_results(
                coco_eval, "bbox", redirect_string,
            )
        return res

    def _derive_coco_results(self, coco_eval, iou_type, summary, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str): specific evaluation task,
                optional values are: "bbox", "segm", "keypoints".
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self.logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        small_table = create_small_table(results)
        self.logger.info("Evaluation results for {}: \n".format(
            iou_type) + small_table)
        if not np.isfinite(sum(results.values())):
            self.logger.info("Note that some metrics cannot be computed.")

        if class_names is None:  # or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = {}
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category[name] = float(ap * 100)
            # results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        table = create_table_with_header(
            results_per_category, headers=["category", "AP"])
        self.logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name,
                       ap in results_per_category.items()})
        if self._dump:
            dump_info_one_task = {
                "task": iou_type,
                "summary": summary.getvalue(),
                "tables": [small_table, table],
            }
            self._dump_infos.append(dump_info_one_task)
        return results


def oi_sgg_evaluation(all_results, predicate_cls_list, post_proc=True):
    logger.info('openimage evaluation: \n')
    result_str = ''

    topk = 100

    # if cfg.TEST.DATASETS[0].find('vg') >= 0:
    #     eval_per_img = True
    #     # eval_per_img = False
    #     prd_k = 1
    # else:
    #     eval_per_img = False
    #     prd_k = 2
    #
    # if cfg.TEST.DATASETS[0].find('oi') >= 0:
    #     eval_ap = True
    # else:
    #     eval_ap = False

    # here we only takes the evaluation option of openimages
    if post_proc:
        prd_k = 2
    else:
        prd_k = 1

    recalls_per_img = {1: [], 5: [], 10: [], 20: [], 50: [], 100: []}
    recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    all_gt_cnt = 0

    topk_dets = []
    for im_i, res in enumerate(tqdm(all_results)):

        # in oi_all_rel some images have no dets
        if res['prd_scores_dist'] is None:
            det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
            det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
            det_labels_s_top = np.zeros(0, dtype=np.int32)
            det_labels_p_top = np.zeros(0, dtype=np.int32)
            det_labels_o_top = np.zeros(0, dtype=np.int32)
            det_scores_top = np.zeros(0, dtype=np.float32)

            det_boxes_so_top = np.zeros((0, 8), dtype=np.float32)
            det_labels_spo_top = np.vstack(
                (det_labels_s_top, det_labels_p_top, det_labels_o_top)).transpose()

            det_scores_top_vis = np.zeros(0, dtype=np.float32)
            if 'prd_scores_bias' in res:
                det_scores_top_bias = np.zeros(0, dtype=np.float32)
            if 'prd_scores_spt' in res:
                det_scores_top_spt = np.zeros(0, dtype=np.float32)
        else:
            det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
            det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
            det_labels_sbj = res['sbj_labels']  # (#num_rel,)
            det_labels_obj = res['obj_labels']  # (#num_rel,)
            det_scores_sbj = res['sbj_scores']  # (#num_rel,)
            det_scores_obj = res['obj_scores']  # (#num_rel,)
            rel_prd_score_dist = res['prd_scores_dist']
            rel_prd_labels = res['prd_rel_label']
            rel_prd_score = res['prd_rel_score']
            rel_trp_prd_scores = res['prd_trp_score']
            pred_rel_pair_idxs = res['pred_rel_pair_idxs']

            if post_proc:
                # take out the predicates classification score
                if 'prd_scores_ttl' in res:
                    rel_det_scores_prd = res['prd_scores_ttl'][:, :]
                else:
                    # N x C (the prediction score of each categories)
                    rel_det_scores_prd = rel_prd_score_dist[:, :]

                rel_det_labels_prd = np.argsort(-rel_det_scores_prd,
                                                axis=1)  # N x C (the prediction labels sort by prediction score) start from 0
                rel_det_labels_prd = rel_det_labels_prd  # rel labels start from 0

                rel_det_scores_prd = -np.sort(-rel_det_scores_prd,
                                              axis=1)  # N x C (the prediction scores sort by prediction score)

                # filtering the results by the productiong of prediction score of subject object and predicates
                det_scores_so = det_scores_sbj * det_scores_obj  # N
                det_scores_spo = det_scores_so[:, None] * \
                    rel_det_scores_prd[:, :prd_k]  # N x prd_K
                # (take top k predicates prediction of each pairs as final prediction,
                # approximation of non-graph constrain setting)

                # det_scores_spo = res['prd_trp_score']
                det_scores_inds = argsort_desc(det_scores_spo)[
                    :topk]  # topk x 2
                # selected the topk score prediction from the N x prd_k predictions
                # first dim:  pair prediction index. second dim: cate id of prediction from this pair

                # take out the correspond tops relationship predation scores and pair boxes and their labels.
                det_scores_top = det_scores_spo[det_scores_inds[:,
                                                                0], det_scores_inds[:, 1]]

                # transforms the selected prediction into the format for evaluation
                det_boxes_so_top = np.hstack(
                    (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
                det_labels_p_top = rel_det_labels_prd[det_scores_inds[:,
                                                                      0], det_scores_inds[:, 1]]
                det_labels_spo_top = np.vstack((det_labels_sbj[det_scores_inds[:, 0]],
                                                det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose()

                # filter the very low prediction scores relationship prediction
                cand_inds = np.where(det_scores_top > 0.000)[0]
                det_boxes_so_top = det_boxes_so_top[cand_inds]
                det_labels_spo_top = det_labels_spo_top[cand_inds]

                det_scores_top = det_scores_top[cand_inds]
                # use the rel pred score as the AP ranking score
                # det_scores_top = rel_det_scores_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]][cand_inds]
            else:

                # non other post process, just use the rank of predictions
                # directly take the topk prediction results
                det_scores_top = rel_trp_prd_scores[:topk]
                det_boxes_so_top = np.hstack((det_boxes_sbj[:topk],
                                              det_boxes_obj[:topk]))
                det_labels_p_top = rel_prd_labels[:topk]  # start from 0

                det_labels_spo_top = np.vstack(
                    (det_labels_sbj[:topk], det_labels_p_top, det_labels_obj[:topk])).transpose()

            det_boxes_s_top = det_boxes_so_top[:, :4]
            det_boxes_o_top = det_boxes_so_top[:, 4:]
            det_labels_s_top = det_labels_spo_top[:, 0]
            det_labels_p_top = det_labels_spo_top[:, 1]
            det_labels_o_top = det_labels_spo_top[:, 2]

        topk_dets.append(dict(image=im_i,
                              det_boxes_s_top=det_boxes_s_top,
                              det_boxes_o_top=det_boxes_o_top,
                              det_labels_s_top=det_labels_s_top,
                              det_labels_p_top=det_labels_p_top,
                              det_labels_o_top=det_labels_o_top,
                              det_scores_top=det_scores_top, )
                         )

        # topk_dets[-1]['det_scores_top_vis'] = det_scores_top_vis
        # if 'prd_scores_bias' in res:
        #     topk_dets[-1]['det_scores_top_bias'] = det_scores_top_bias
        # if 'prd_scores_spt' in res:
        #     topk_dets[-1]['det_scores_top_spt'] = det_scores_top_spt

        gt_boxes_sbj = res['gt_sbj_boxes']  # (#num_gt, 4)
        gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
        gt_labels_sbj = res['gt_sbj_labels']  # (#num_gt,)
        gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
        gt_labels_prd = res['gt_prd_labels']  # start from 0 (#num_gt,)
        gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
        gt_labels_spo = np.vstack(
            (gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
        # Compute recall. It's most efficient to match once and then do recall after
        # det_boxes_so_top is (#num_rel, 8)
        # det_labels_spo_top is (#num_rel, 3)
        pred_to_gt = _compute_pred_matches(
            gt_labels_spo, det_labels_spo_top,
            gt_boxes_so, det_boxes_so_top)

        ###########################################
        # 对fg做点扰动 把retrive回来的fg的分数拉起来
        # for pred_id, each_pred_mat in enumerate(pred_to_gt):
        #     if len(each_pred_mat) > 0:
        #         thres = topk_dets[-1]['det_scores_top'][11]
        #         re_rank = np.mean(topk_dets[-1]['det_scores_top'][:8])
        #         print(thres, re_rank)
        #         if topk_dets[-1]['det_scores_top'][pred_id] < thres:
        #             topk_dets[-1]['det_scores_top'][pred_id] = re_rank
        ###########################################

        # perimage recall
        for k in recalls_per_img:
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
            else:
                match = []
            # in case there is no gt
            rec_i = float(len(match)) / float(gt_labels_spo.shape[0] + 1e-12)
            recalls_per_img[k].append(rec_i)

        # all dataset recall
        all_gt_cnt += gt_labels_spo.shape[0]
        for k in recalls:
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
            else:
                match = []
            recalls[k] += len(match)

        topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                  gt_boxes_obj=gt_boxes_obj,
                                  gt_labels_sbj=gt_labels_sbj,
                                  gt_labels_obj=gt_labels_obj,
                                  gt_labels_prd=gt_labels_prd))

    # remove the background categoires
    predicate_cls_list_woBG = predicate_cls_list
    for k in recalls_per_img.keys():
        recalls_per_img[k] = np.mean(recalls_per_img[k])

    for k in recalls:
        recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)

    # prepare dets for each class
    logger.info('Preparing dets for mAP...')
    cls_image_ids, cls_dets, cls_gts, npos = prepare_mAP_dets(
        topk_dets, len(predicate_cls_list_woBG))
    all_npos = sum(npos)

    assert all_npos == np.concatenate(
        [each['gt_labels_prd'] for each in topk_dets]).shape[0]

    rel_mAP = 0.
    w_rel_mAP = 0.
    ap_str = ''
    per_class_res = ''
    empty_cate = ''
    for c in tqdm(range(len(predicate_cls_list_woBG))):
        rec, prec, ap = ap_eval(
            cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], True)

        if ap is None:
            empty_cate += f"{c}-{predicate_cls_list_woBG[c]}, "
            continue

        if len(rec) == 0:
            rec = [0.0]

        weighted_ap = ap * float(npos[c]) / float(all_npos)
        w_rel_mAP += weighted_ap
        rel_mAP += ap
        ap_str += '{:.2f}, '.format(100 * ap)
        per_class_res += '{}: {:.3f} / {:.3f} / {:.3f} ({:.6f}:{}/{}), '.format(
            predicate_cls_list_woBG[c], 100 * ap, 100 *
            weighted_ap, rec[-1] * 100, float(npos[c]) / float(all_npos),
            npos[c], all_npos)

    rel_mAP /= len(predicate_cls_list_woBG)
    logger.info(f"Not gt instance in categories: {empty_cate}")
    result_str += '\nrel mAP: {:.2f}, weighted rel mAP: {:.2f}\n'.format(
        100 * rel_mAP, 100 * w_rel_mAP)
    result_str += 'rel AP perclass: AP/ weighted-AP / recall (weight-total_fg_propotion)\n'
    result_str += per_class_res + "\n\n"
    phr_mAP = 0.
    w_phr_mAP = 0.
    ap_str = ''

    per_class_res = ''
    for c in range(len(predicate_cls_list_woBG)):
        rec, prec, ap = ap_eval(
            cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], False)

        if ap is None:
            continue
        if len(rec) == 0:
            rec = [0.0]

        weighted_ap = ap * float(npos[c]) / float(all_npos)
        w_phr_mAP += weighted_ap
        phr_mAP += ap
        ap_str += '{:.2f}, '.format(100 * ap)
        per_class_res += '{}: {:.3f} / {:.3f} / {:.3f} ({:.6f}:{}/{}), '.format(
            predicate_cls_list_woBG[c], 100 * ap, 100 *
            weighted_ap, rec[-1] * 100, float(npos[c]) / float(all_npos),
            npos[c], all_npos)

    phr_mAP /= len(predicate_cls_list_woBG)
    result_str += '\nphr mAP: {:.2f}, weighted phr mAP: {:.2f}\n'.format(
        100 * phr_mAP, 100 * w_phr_mAP)
    result_str += 'rel AP perclass: AP/ weighted-AP / recall (weight-total_fg_propotion)\n'
    result_str += per_class_res + "\n\n"

    # total: 0.4 x rel_mAP + 0.2 x R@50 + 0.4 x phr_mAP
    final_score = 0.4 * rel_mAP + 0.2 * recalls[50] + 0.4 * phr_mAP

    # total: 0.4 x w_rel_mAP + 0.2 x R@50 + 0.4 x w_phr_mAP
    w_final_score = 0.4 * w_rel_mAP + 0.2 * recalls[50] + 0.4 * w_phr_mAP
    result_str += "recall@50: {:.2f}, recall@100: {:.2f}\n".format(
        100 * recalls[50], 100 * recalls[100])
    result_str += "recall@50: {:.2f}, recall@100: {:.2f} (per images)\n\n".format(100 * recalls_per_img[50],
                                                                                  100 * recalls_per_img[100])

    result_str += "weighted_res: 0.4 * w_rel_mAP + 0.2 * recall@50 + 0.4 * w_phr_mAP \n"
    result_str += 'final_score:{:.2f}  weighted final_score: {:.2f}\n'.format(
        final_score * 100, w_final_score * 100)

    res_dict = dict(
        mAP_rel=rel_mAP,
        wmAP_rel=w_rel_mAP,
        mAP_phr=phr_mAP,
        wmAP_phr=w_phr_mAP,
        R50=recalls[50],
        final_score=final_score,
        w_final_score=w_final_score,
    )

    result_str += "=" * 80
    result_str += "\n\n"

    logger.info('Done.')

    # logger.info(result_str)

    return result_str, res_dict


def adapt_results(groudtruths, predictions, ):
    packed_results = []
    for gt, pred in zip(groudtruths, predictions):
        gt = copy.deepcopy(gt)
        pred = copy.deepcopy(pred)

        pred_boxlist = pred.convert('xyxy').to("cpu")
        pred_ent_scores = pred_boxlist.get_field('pred_scores').detach().cpu()
        pred_ent_labels = pred_boxlist.get_field(
            'pred_labels').long().detach().cpu()
        # pred_ent_labels = pred_ent_labels - 1  # remove the background class

        pred_rel_pairs = pred_boxlist.get_field(
            'rel_pair_idxs').long().detach().cpu()  # N * R * 2
        pred_rel_score_dist = pred_boxlist.get_field(
            'pred_rel_dist').detach().cpu()  # N * C
        pred_rel_labels = pred_boxlist.get_field(
            'pred_rel_label').detach().cpu()
        pred_rel_score = pred_boxlist.get_field(
            'pred_rel_score').detach().cpu()

        prd_trp_score = None
        if pred_boxlist.has_field('pred_rel_trp_score'):
            prd_trp_score = pred_boxlist.get_field(
                'pred_rel_trp_score').detach().cpu().numpy()

        sbj_boxes = pred_boxlist.bbox[pred_rel_pairs[:, 0], :].numpy()
        sbj_labels = pred_ent_labels[pred_rel_pairs[:, 0]].numpy()
        sbj_scores = pred_ent_scores[pred_rel_pairs[:, 0]].numpy()

        obj_boxes = pred_boxlist.bbox[pred_rel_pairs[:, 1], :].numpy()
        obj_labels = pred_ent_labels[pred_rel_pairs[:, 1]].numpy()
        obj_scores = pred_ent_scores[pred_rel_pairs[:, 1]].numpy()

        gt_boxlist = gt.convert('xyxy').to("cpu")
        gt_ent_labels = gt_boxlist.get_field('labels')

        gt_rel_tuple = gt_boxlist.get_field(
            'relation_tuple').long().detach().cpu()
        sbj_gt_boxes = gt_boxlist.bbox[gt_rel_tuple[:, 0], :].detach(
        ).cpu().numpy()
        obj_gt_boxes = gt_boxlist.bbox[gt_rel_tuple[:, 1], :].detach(
        ).cpu().numpy()
        sbj_gt_classes = gt_ent_labels[gt_rel_tuple[:, 0]].long(
        ).detach().cpu().numpy()
        obj_gt_classes = gt_ent_labels[gt_rel_tuple[:, 1]].long(
        ).detach().cpu().numpy()
        prd_gt_classes = gt_rel_tuple[:, -1].long().detach().cpu().numpy()

        # align GT class from 0
        # prd_gt_classes = prd_gt_classes - 1
        # gt_rel_tuple[:, -1] -= 1

        return_dict = dict(sbj_boxes=sbj_boxes,
                           sbj_labels=sbj_labels.astype(np.int32, copy=False),
                           sbj_scores=sbj_scores,
                           obj_boxes=obj_boxes,
                           obj_labels=obj_labels.astype(np.int32, copy=False),
                           obj_scores=obj_scores,
                           prd_scores_dist=pred_rel_score_dist,
                           prd_trp_score=prd_trp_score,
                           prd_rel_label=pred_rel_labels,
                           prd_rel_score=pred_rel_score,
                           pred_rel_pair_idxs=pred_rel_pairs,
                           # prd_scores_bias=prd_scores,
                           # prd_scores_spt=prd_scores,
                           # prd_ttl_scores=prd_scores,
                           gt_sbj_boxes=sbj_gt_boxes,
                           gt_obj_boxes=obj_gt_boxes,
                           gt_sbj_labels=sbj_gt_classes.astype(
                               np.int32, copy=False),
                           gt_obj_labels=obj_gt_classes.astype(
                               np.int32, copy=False),
                           gt_prd_labels=prd_gt_classes.astype(np.int32, copy=False))

        packed_results.append(return_dict)

    return packed_results


def prepare_mAP_dets(topk_dets, cls_num):
    cls_image_ids = [[] for _ in range(cls_num)]
    cls_dets = [{'confidence': np.empty(0),
                 'BB_s': np.empty((0, 4)),
                 'BB_o': np.empty((0, 4)),
                 'BB_r': np.empty((0, 4)),
                 'LBL_s': np.empty(0),
                 'LBL_o': np.empty(0)} for _ in range(cls_num)]
    cls_gts = [{} for _ in range(cls_num)]
    npos = [0 for _ in range(cls_num)]
    for dets in topk_dets:
        image_id = dets['image']
        sbj_boxes = dets['det_boxes_s_top']
        obj_boxes = dets['det_boxes_o_top']
        rel_boxes = boxes_union(sbj_boxes, obj_boxes)
        sbj_labels = dets['det_labels_s_top']
        obj_labels = dets['det_labels_o_top']
        prd_labels = dets['det_labels_p_top']
        det_scores = dets['det_scores_top']
        gt_boxes_sbj = dets['gt_boxes_sbj']
        gt_boxes_obj = dets['gt_boxes_obj']
        gt_boxes_rel = boxes_union(gt_boxes_sbj, gt_boxes_obj)
        gt_labels_sbj = dets['gt_labels_sbj']
        gt_labels_prd = dets['gt_labels_prd']
        gt_labels_obj = dets['gt_labels_obj']
        for cls_id in range(cls_num):
            cls_inds = np.where(prd_labels == (cls_id))[0]
            if len(cls_inds):
                cls_sbj_boxes = sbj_boxes[cls_inds]
                cls_obj_boxes = obj_boxes[cls_inds]
                cls_rel_boxes = rel_boxes[cls_inds]
                cls_sbj_labels = sbj_labels[cls_inds]
                cls_obj_labels = obj_labels[cls_inds]
                cls_det_scores = det_scores[cls_inds]
                cls_dets[cls_id]['confidence'] = np.concatenate(
                    (cls_dets[cls_id]['confidence'], cls_det_scores))
                cls_dets[cls_id]['BB_s'] = np.concatenate(
                    (cls_dets[cls_id]['BB_s'], cls_sbj_boxes), 0)
                cls_dets[cls_id]['BB_o'] = np.concatenate(
                    (cls_dets[cls_id]['BB_o'], cls_obj_boxes), 0)
                cls_dets[cls_id]['BB_r'] = np.concatenate(
                    (cls_dets[cls_id]['BB_r'], cls_rel_boxes), 0)
                cls_dets[cls_id]['LBL_s'] = np.concatenate(
                    (cls_dets[cls_id]['LBL_s'], cls_sbj_labels))
                cls_dets[cls_id]['LBL_o'] = np.concatenate(
                    (cls_dets[cls_id]['LBL_o'], cls_obj_labels))
                cls_image_ids[cls_id] += [image_id] * len(cls_inds)
            cls_gt_inds = np.where(gt_labels_prd == cls_id)[0]
            cls_gt_boxes_sbj = gt_boxes_sbj[cls_gt_inds]
            cls_gt_boxes_obj = gt_boxes_obj[cls_gt_inds]
            cls_gt_boxes_rel = gt_boxes_rel[cls_gt_inds]
            cls_gt_labels_sbj = gt_labels_sbj[cls_gt_inds]
            cls_gt_labels_obj = gt_labels_obj[cls_gt_inds]
            cls_gt_num = len(cls_gt_inds)
            det = [False] * cls_gt_num
            npos[cls_id] = npos[cls_id] + cls_gt_num
            # N x 4, The N same categories relationship in image of image_id
            cls_gts[cls_id][image_id] = {'gt_boxes_sbj': cls_gt_boxes_sbj,
                                         'gt_boxes_obj': cls_gt_boxes_obj,
                                         'gt_boxes_rel': cls_gt_boxes_rel,
                                         'gt_labels_sbj': cls_gt_labels_sbj,
                                         'gt_labels_obj': cls_gt_labels_obj,
                                         'gt_num': cls_gt_num,
                                         'det': det}
    return cls_image_ids, cls_dets, cls_gts, npos


def get_ap(rec, prec):
    """Compute AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    ii = np.where(mrec[1:] != mrec[:-1])[0]

    # print(i, mpre[i])
    # print(mpre, mrec)
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii])
    return ap


def ap_eval(image_ids,
            dets,
            gts,
            npos,
            rel_or_phr=True,
            ovthresh=0.5):
    """
    Top level function that does the relationship AP evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    confidence = dets['confidence']
    BB_s = dets['BB_s']
    BB_o = dets['BB_o']
    BB_r = dets['BB_r']
    LBL_s = dets['LBL_s']
    LBL_o = dets['LBL_o']

    # sort by confidence in descending order
    sorted_ind = np.argsort(confidence, kind="mergesort")[::-1]
    BB_s = BB_s[sorted_ind, :]
    BB_o = BB_o[sorted_ind, :]
    BB_r = BB_r[sorted_ind, :]
    LBL_s = LBL_s[sorted_ind]
    LBL_o = LBL_o[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    num_detection = len(image_ids)

    tp = np.zeros(num_detection)
    fp = np.zeros(num_detection)

    # mark the gt whether hit by other prediciton
    gt_hit_book = {k: [False] * v['gt_num'] for k, v in gts.items()}
    # check
    y_test = np.zeros(num_detection)
    y_pred = confidence[sorted_ind]

    repeat_hit_num = 0
    for rank_idx in range(num_detection):
        gt_rel = gts[image_ids[rank_idx]]
        visited = gt_hit_book[image_ids[rank_idx]]
        bb_s = BB_s[rank_idx, :].astype(float)
        bb_o = BB_o[rank_idx, :].astype(float)
        bb_r = BB_r[rank_idx, :].astype(float)
        lbl_s = LBL_s[rank_idx]
        lbl_o = LBL_o[rank_idx]
        ovmax = -np.inf
        BBGT_s = gt_rel['gt_boxes_sbj'].astype(float)
        BBGT_o = gt_rel['gt_boxes_obj'].astype(float)
        BBGT_r = gt_rel['gt_boxes_rel'].astype(float)
        LBLGT_s = gt_rel['gt_labels_sbj']
        LBLGT_o = gt_rel['gt_labels_obj']
        if BBGT_s.size > 0:
            valid_mask = np.logical_and(LBLGT_s == lbl_s, LBLGT_o == lbl_o)
            if valid_mask.any():
                if rel_or_phr:  # means it is evaluating relationships
                    # 1 x num_gt
                    overlaps_s = bbox_overlaps(
                        bb_s[None, :].astype(dtype=np.float32, copy=False),
                        BBGT_s.astype(dtype=np.float32, copy=False))[0]
                    overlaps_o = bbox_overlaps(
                        bb_o[None, :].astype(dtype=np.float32, copy=False),
                        BBGT_o.astype(dtype=np.float32, copy=False))[0]
                    overlaps = np.minimum(overlaps_s, overlaps_o)
                else:
                    overlaps = bbox_overlaps(
                        bb_r[None, :].astype(dtype=np.float32, copy=False),
                        BBGT_r.astype(dtype=np.float32, copy=False))[0]
                overlaps *= valid_mask
                # find the best matching relatioships
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            else:
                ovmax = 0.
                jmax = -1

        if ovmax > ovthresh:
            assert jmax >= 0
            if not visited[jmax]:
                # tp.append(1.)
                # fp.append(0.)
                tp[rank_idx] = 1.
                visited[jmax] = 1
            else:
                # fp.append(1.)
                # tp.append(0.)
                repeat_hit_num += 1
                fp[rank_idx] = 1.

        else:
            # fp.append(1.)
            # tp.append(0.)
            fp[rank_idx] = 1.
            y_test[rank_idx] = 0

    # add missed_gt
    missed_gt_num = 0
    y_test = y_test.tolist()
    y_pred = y_pred.tolist()
    for k, hit_flag_book in gt_hit_book.items():
        for is_hit_flag in hit_flag_book:
            if not is_hit_flag:
                missed_gt_num += 1
                y_test.append(1)
                y_pred.append(0.0)
                # tp = np.concatenate((tp, [1]))
                # fp = np.concatenate((fp, [0]))

    # compute precision recall
    if npos > 0:
        acc_FP = np.cumsum(fp)
        acc_TP = np.cumsum(tp)

        # ground truth
        rec = acc_TP / (npos + 1e-12)
        prec = acc_TP / np.maximum(acc_TP + acc_FP, np.finfo(np.float64).eps)

        ap = get_ap(rec, prec)
        # print(repeat_hit_num)
        return rec, prec, ap

    # no prediction or gt from this categories, can not calculate any metrics
    return None, None, None


def boxes_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.minimum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.maximum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()


def _compute_pred_matches(gt_triplets, pred_triplets,
                          gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh: Do y
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate(
                (gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate(
                (box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            gt_box_union = gt_box_union.astype(dtype=np.float32, copy=False)
            box_union = box_union.astype(dtype=np.float32, copy=False)
            inds = bbox_overlaps(gt_box_union[None],
                                 box_union=box_union)[0] >= iou_thresh

        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
