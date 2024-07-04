import contextlib
import copy
import io
import json
import logging
import os
import pickle
from collections import OrderedDict, defaultdict

import numpy as np
import tabulate
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import itertools

from .comm import get_rank

from .boxlist import BoxList
from .vg_sgg_eval_tools import SGRecall, SGNoGraphConstraintRecall, SGPairAccuracy, SGMeanRecall, SGRelVecRecall, \
    SGZeroShotRecall, \
    SGStagewiseRecall, SGNGMeanRecall

logger = logging.getLogger("lavis." + __name__)


topk_range = [20, 50, 100, 200, 300, 400, 999]
class VisualGenomeSGGEvaluator():
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, meta, eval_rel, output_dir=None,tasks=None):

        self._results = OrderedDict()
        self._predictions = OrderedDict()
        self._groundtruths = OrderedDict()
        self._predictions_tmp = OrderedDict()

        self._tasks = tasks
        self._output_dir = output_dir
        self._metadata = meta

        self._cpu_device = torch.device("cpu")

        self.eval_rel = eval_rel

        self.dump_idx = 0
        self._dump_infos = []

    # def _tasks_from_config(self, cfg):
    #     mode_list = ['bbox']

    #     if self.eval_rel:
    #         if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
    #             if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
    #                 rel_mode = 'predcls'
    #             else:
    #                 rel_mode = 'sgcls'
    #         else:
    #             rel_mode = 'sgdet'

    #         mode_list.append(rel_mode)

    #     return mode_list

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = OrderedDict()
        self._groundtruths = OrderedDict()

    def process(self, inputs, outputs):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            in VG dataset, we have:
                dataset_dict = {
                    'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
                    "instances": target,
                    "relationships": relationships,
                    "file_name": self.filenames[index],
                    'image_id': index,
                    'height': image.shape[0],
                    'width': image.shape[1]
                }
            output: dump things back to the BoxList form ientical with the
                    initial SGG codebase for adapt the evaluation protocol
        """

        # transforms the data into the required format
        for input, output in zip(inputs, outputs):
            # assert  output["width"] == input["width"] and  input["height"] == output["height"]
            image_id = input['image_id']

            image_width = input["width"]  # the x axis of image
            image_height = input["height"]  # the y axis of images
            image_size = (image_width, image_height)
            bbox = input["instances"].gt_boxes.tensor
            groundtruths = BoxList(bbox, image_size, mode="xyxy")
            groundtruths.add_field('labels', input["instances"].gt_classes)

            if self.eval_rel:
                gt_relationships = input["relationships"]
                groundtruths.add_field('relation_tuple', gt_relationships.relation_tuple)

            self._groundtruths[image_id] = groundtruths.to(self._cpu_device)

            if self.eval_rel:
                pred_instances = output["relationships"].instances
            else:
                pred_instances = output["instances"]

            pred_instances = output["instances"]

            image_height, image_width = pred_instances.image_size
            image_size = (image_width, image_height)
            bbox = pred_instances.pred_boxes.tensor
            prediction = BoxList(bbox, image_size, mode="xyxy")
            prediction.add_field('pred_labels', pred_instances.pred_classes)
            prediction.add_field('pred_scores', pred_instances.scores)
            prediction.add_field('pred_score_dist', pred_instances.pred_score_dist)
            prediction.add_field('image_id', image_id)

            if self.eval_rel:
                pred_relationships = output["relationships"]
                # obtain the related relationships predictions attributes
                prediction.add_field('rel_pair_idxs', pred_relationships.rel_pair_tensor)
                prediction.add_field('pred_rel_dist', pred_relationships.pred_rel_dist)
                prediction.add_field('pred_rel_score', pred_relationships.pred_rel_scores)
                prediction.add_field('pred_rel_label', pred_relationships.pred_rel_classs)

                if pred_relationships.has('pred_rel_trp_scores'):
                    prediction.add_field('pred_rel_trp_score', pred_relationships.pred_rel_trp_scores)

                if pred_relationships.has('pred_rel_confidence'):
                    prediction.add_field('rel_confidence', pred_relationships.pred_rel_confidence)

                if pred_relationships.has('rel_vec'):
                    prediction.add_field('rel_vec', pred_relationships.rel_vec)



            self._predictions[image_id] = prediction.to(self._cpu_device)
            self._predictions_tmp[image_id] = prediction.to(self._cpu_device)


    def evaluate(self):
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

        # dump the remaining results 

        if self._dump and self._output_dir is not None:
            out_dir = os.path.join(self._output_dir, "inference_new", self.dataset_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            file_path = os.path.join(out_dir, f"inference_prediction{get_rank()}-{self.dump_idx}.pkl")
            print("prediction is saving to", file_path)
            torch.save(self._predictions_tmp, file_path)
            self.dump_idx += 1
            self._predictions_tmp = OrderedDict()

        eval_types = ['bbox']

        self.chunk_gather()

        # only main process do evaluation
        # return empty for following procedure
        if get_rank() != 0:
            return {}

        if self.eval_rel:
            eval_types.append('relation')

        # evaluate the entities_box

        predictions = self._predictions
        groundtruths = self._groundtruths

        if self._dump and self._output_dir is not None:
            out_dir = os.path.join(self._output_dir, "inference", self.dataset_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            file_path = os.path.join(out_dir, "inference_prediction.pkl")
            print("prediction is saving to", file_path)
            torch.save(predictions, file_path)

        if "bbox" in eval_types:
            # create a Coco-like object that we can use to evaluate detection!
            for longtail_set in [
                # [31, 20, 48, 30, 22, 29, 8, 50, 21, 1, 43, 49, 40, 23, 38, 41],
                #                 [6, 7, 33, 11, 46, 16, 47, 25, 19, 5, 9, 35, 24, 10, 4, 14, 13],
                #                 [12, 36, 44, 42, 32, 2, 45, 28, 26, 3, 17, 18, 34, 37, 27, 39, 15],
                None]:
                ############################################################
                # box localization evaluation
                anns = []
                # prepare GTs
                for image_id, gt in enumerate(groundtruths):
                    ent_id = set()
                    if longtail_set is not None:
                        for each in gt.get_field('relation_tuple').tolist():
                            # selected entity categories
                            if each[-1] in longtail_set:
                                ent_id.add(each[0])
                                ent_id.add(each[1])
                    else:
                        ent_id = range(len(gt.get_field('labels')))

                    ent_id = torch.Tensor(list(ent_id)).long()
                    # set all gt label as one
                    labels = [1 for _ in gt.get_field('labels')[ent_id]]

                    boxes = gt.bbox[ent_id].tolist()  # xyxy
                    for cls, box in zip(labels, boxes):
                        anns.append({
                            'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                            'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                            'category_id': cls,
                            'id': len(anns),
                            'image_id': image_id,
                            'iscrowd': 0,
                        })
                fauxcoco = COCO()

                fauxcoco.dataset = {
                    'info': {'description': 'use coco script for vg detection evaluation'},
                    'images': [{'id': i} for i in range(len(groundtruths))],
                    'categories': [
                        {'supercategory': 'person', 'id': i, 'name': name}
                        for i, name in enumerate(self._metadata.ind_to_classes) if name != '__background__'
                    ],
                    'annotations': anns,
                }
                fauxcoco.createIndex()

                # format predictions to coco-like
                cocolike_predictions = []
                for image_id, prediction in enumerate(predictions):
                    box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
                    score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
                    label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
                    label = np.array([1 for each in label])
                    # for predcls, we set label and score to groundtruth
                    image_id = np.asarray([image_id] * len(box))
                    cocolike_predictions.append(
                        np.column_stack((image_id, box, score, label))
                    )
                    # logger.info(cocolike_predictions)
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

                logger.info(f"The entity box localization performance:\n{coco_summary}")

                res = self._derive_coco_results(
                    coco_eval, "bbox", redirect_string,
                    class_names=self._metadata.ind_to_classes[:-1], 
                )
                self._results["bbox_loc"] = res

                ############################################################

                anns = []
                for image_id, gt in enumerate(groundtruths):
                    ent_id = set()
                    if longtail_set is not None:
                        for each in gt.get_field('relation_tuple').tolist():
                            # selected entity categories
                            if each[-1] in longtail_set:
                                ent_id.add(each[0])
                                ent_id.add(each[1])
                    else:
                        ent_id = range(len(gt.get_field('labels')))

                    ent_id = torch.Tensor(list(ent_id)).long()
                    labels = gt.get_field('labels')[ent_id].tolist()  # integer
                    boxes = gt.bbox[ent_id].tolist()  # xyxy
                    for cls, box in zip(labels, boxes):
                        anns.append({
                            'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                            'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                            'category_id': cls,
                            'id': len(anns),
                            'image_id': image_id,
                            'iscrowd': 0,
                        })
                fauxcoco = COCO()

                fauxcoco.dataset = {
                    'info': {'description': 'use coco script for vg detection evaluation'},
                    'images': [{'id': i} for i in range(len(groundtruths))],
                    'categories': [
                        {'supercategory': 'person', 'id': i, 'name': name}
                        for i, name in enumerate(self._metadata.ind_to_classes) if name != '__background__'
                    ],
                    'annotations': anns,
                }
                fauxcoco.createIndex()

                # format predictions to coco-like
                cocolike_predictions = []
                for image_id, prediction in enumerate(predictions):
                    box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
                    score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
                    label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
                    # for predcls, we set label and score to groundtruth
                    if 'predcls' in self._tasks:
                        label = prediction.get_field('labels').detach().cpu().numpy()
                        score = np.ones(label.shape[0])
                        assert len(label) == len(box)
                    image_id = np.asarray([image_id] * len(box))
                    cocolike_predictions.append(
                        np.column_stack((image_id, box, score, label))
                    )
                    # logger.info(cocolike_predictions)
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

                res = self._derive_coco_results(
                    coco_eval, "bbox", redirect_string,
                    class_names=self._metadata.ind_to_classes[:-1],
                    longtail_part_eval=True,
                )
                self._results["bbox"] = res

                logger.info(f"The entity detection:\n{coco_summary}")
                ############################################################

        if 'relation' in eval_types:
            predicates_categories = self._metadata.ind_to_predicates
            _, vg_sgg_eval_res = classic_vg_sgg_evaluation(predictions,
                                                           groundtruths,
                                                           predicates_categories)

            self._results['sgg_vg_metrics'] = vg_sgg_eval_res

        return copy.deepcopy(self._results)

    def _derive_coco_results(self, coco_eval, iou_type, summary, class_names=None, longtail_part_eval=False):
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
            logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        small_table = create_small_table(results)
        logger.info("Evaluation results for {}: \n".format(iou_type) + small_table)
        if not np.isfinite(sum(results.values())):
            logger.info("Note that some metrics cannot be computed.")

        if class_names is None:  # or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        recalls = coco_eval.eval["recall"]

        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        if longtail_part_eval:
            try:
                ent_longtail_part = ENTITY_LONGTAIL_DICT
                results_per_category = {}
                recall_per_category = {}
                long_tail_part_res = defaultdict(list)
                long_tail_part_res_recall = defaultdict(list)
                results_per_category_show = {}

                for idx, name in enumerate(class_names):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]

                    recall = recalls[:, idx, 0, -1]
                    recall = recall[recall > -1]

                    if precision.size:
                        ap = np.mean(precision)
                        recall = np.mean(recall)
                        long_tail_part_res[ent_longtail_part[idx]].append(ap)
                        long_tail_part_res_recall[ent_longtail_part[idx]].append(recall)
                    else:
                        ap = float("nan")
                        recall = float("nan")
                        long_tail_part_res[ent_longtail_part[idx]].append(0)
                        long_tail_part_res_recall[ent_longtail_part[idx]].append(0)

                    results_per_category_show[f'{name} {ent_longtail_part[idx]}'] = float(ap * 100)

                    results_per_category[name] = float(ap * 100)
                    recall_per_category[name] = float(recall * 100)

                (fig,
                longtail_part_performance) = self.per_class_performance_dump(results_per_category,
                                                                            long_tail_part_res)
                # tabulate it
                table = create_table_with_header(results_per_category_show, headers=["category", "AP"])
                logger.info("Per-category {} AP: \n".format(iou_type) + table)

                table = create_table_with_header(recall_per_category, headers=["category", "recall"])
                logger.info("Per-category {} recall: \n".format(iou_type) + table)

                save_file = os.path.join(self.cfg.OUTPUT_DIR, f"ent_ap_per_cls.png")
                fig.savefig(save_file, dpi=300)
                logger.info("Longtail part {} AP: \n".format(iou_type) + longtail_part_performance)

                (fig,
                longtail_part_performance) = self.per_class_performance_dump(recall_per_category,
                                                                            long_tail_part_res_recall)
                save_file = os.path.join(self.cfg.OUTPUT_DIR, f"ent_recall_per_cls.png")
                fig.savefig(save_file, dpi=300)
                logger.info("Longtail part {} recall: \n".format(iou_type) + longtail_part_performance)

                results.update({"AP-" + name: ap for name, ap in results_per_category.items()})
            except:
                pass

        return results

    def per_class_performance_dump(self, results_per_category, long_tail_part_res):
        ent_sorted_cls_list = self.cfg.DATASETS.ENTITY_SORTED_CLS_LIST
        cate_names = []
        ap_sorted = []
        for i in ent_sorted_cls_list:
            cate_name = self._metadata.ind_to_classes[i]
            cate_names.append(cate_name)

            if results_per_category.get(cate_name) is not None:
                ap_sorted.append(results_per_category[cate_name])
            else:
                ap_sorted.append(0)

        fig, axs_c = plt.subplots(1, 1, figsize=(18, 5), tight_layout=True)
        fig.set_facecolor((1, 1, 1))
        axs_c.bar(cate_names, ap_sorted, width=0.6, zorder=0)
        axs_c.grid()
        plt.sca(axs_c)
        plt.xticks(rotation=-90, )

        longtail_part_performance = ''
        for k, name in zip(['h', 'b', 't'], ['head', 'body', 'tail']):
            longtail_part_performance += f'{name}: {np.mean(long_tail_part_res[k]) * 100:.2f}; '

        return fig, longtail_part_performance


def classic_vg_sgg_evaluation(
        predictions,
        groundtruths,
        entities_categories: list,
        predicates_categories: list,
        zeroshot_triplet_dir="/mnt/petrelfs/lirongjie/project/cvpods/datasets/vg/vg_motif_anno/zeroshot_triplet.pytorch",
        zeroshot_predicate=None,
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load(
        zeroshot_triplet_dir,
        map_location=torch.device("cpu")).long().numpy()

    attribute_on = False
    num_attributes = 1

    mode = 'sgdet'
    num_rel_category = len(predicates_categories) + 1
    num_ent_category = len(entities_categories) + 1

    iou_thres = 0.5
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    avg_metrics = 0
    result_str = '\n' + '=' * 100 + '\n'

    result_dict = {}
    result_dict_list_to_log = []

    result_str = '\n'
    evaluator = {}
    rel_eval_result_dict = {}
    # tradictional Recall@K
    eval_recall = SGRecall(rel_eval_result_dict)
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall

    eval_recall_phrdet = SGRecall(rel_eval_result_dict, )
    eval_recall_phrdet.register_container('phrdet')
    evaluator['eval_recall_phrdet'] = eval_recall_phrdet

    # no graphical constraint
    eval_nog_recall = SGNoGraphConstraintRecall(rel_eval_result_dict, entities_categories, predicates_categories,)
    eval_nog_recall.register_container(mode)
    evaluator['eval_nog_recall'] = eval_nog_recall

    # test on different distribution
    eval_zeroshot_recall = SGZeroShotRecall(rel_eval_result_dict)
    eval_zeroshot_recall.register_container(mode)
    evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

    # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
    eval_pair_accuracy = SGPairAccuracy(rel_eval_result_dict)
    eval_pair_accuracy.register_container(mode)
    evaluator['eval_pair_accuracy'] = eval_pair_accuracy

    # used for meanRecall@K
    eval_mean_recall = SGMeanRecall(rel_eval_result_dict, entities_categories, predicates_categories,
                                    print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall

    eval_mean_recall_phrdet = SGMeanRecall(rel_eval_result_dict, entities_categories, predicates_categories,
                                           print_detail=True)
    eval_mean_recall_phrdet.register_container('phrdet')
    evaluator['eval_mean_recall_phrdet'] = eval_mean_recall_phrdet

    # used for NG-meanRecall@K
    eval_ng_mean_recall = SGNGMeanRecall(result_dict, entities_categories, predicates_categories,
                                         print_detail=True)
    eval_ng_mean_recall.register_container(mode)
    evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

    eval_stagewise_recall = SGStagewiseRecall(entities_categories, predicates_categories, rel_eval_result_dict)
    eval_stagewise_recall.register_container(mode)
    evaluator['eval_stagewise_recall'] = eval_stagewise_recall

    # eval_rel_vec_recall = SGRelVecRecall(cfg, result_dict, predicates_categories)
    # eval_rel_vec_recall.register_container(mode)
    # evaluator['eval_rel_vec_recall'] = eval_rel_vec_recall

    # prepare all inputs
    global_container = {}
    global_container['zeroshot_triplet'] = zeroshot_triplet
    global_container['result_dict'] = rel_eval_result_dict
    global_container['mode'] = mode
    # global_container['multiple_preds'] = multiple_preds
    global_container['num_rel_category'] = num_rel_category
    global_container['iou_thres'] = iou_thres
    global_container['attribute_on'] = attribute_on
    global_container['num_attributes'] = num_attributes

    indexing_acc = defaultdict(list)
    logger.info("evaluating relationship predictions..")
    for groundtruth, prediction in tqdm(zip(groundtruths, predictions), total=len(predictions)):
        evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

        if prediction.has_field("ent_idx_acc_top1"):
            for k in prediction.fields():
                if 'acc' in k:
                    v = prediction.get_field(k)
                    if not torch.isnan(v).any():
                        indexing_acc[k].append(v.item())

    # calculate mean recall
    eval_mean_recall.calculate_mean_recall(mode)
    eval_ng_mean_recall.calculate_mean_recall(mode)
    eval_mean_recall_phrdet.calculate_mean_recall('phrdet')

    def generate_eval_res_dict(evaluator, mode):
        res_dict = {}
        for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
            res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
        return res_dict

    def longtail_part_eval(evaluator, mode):
        longtail_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        assert "mean_recall" in evaluator.type
        res_dict = {}
        res_str = "\nlongtail part recall:\n"
        for topk, cate_rec_list in evaluator.result_dict[f'{mode}_{evaluator.type}_list'].items():
            part_recall = {"h": [], "b": [], "t": [], }
            for idx, each_cat_recall in enumerate(cate_rec_list):
                part_recall[longtail_part_dict[idx + 1]].append(each_cat_recall)
            res_dict[f"sgdet_longtail_part_recall/top{topk}/head"] = np.mean(part_recall['h'])
            res_dict[f"sgdet_longtail_part_recall/top{topk}/body"] = np.mean(part_recall['b'])
            res_dict[f"sgdet_longtail_part_recall/top{topk}/tail"] = np.mean(part_recall['t'])
            res_str += f"Top{topk:4}: head: {np.mean(part_recall['h']):.4f} " \
                       f"body: {np.mean(part_recall['b']):.4f} " \
                       f"tail: {np.mean(part_recall['t']):.4f}\n"

        return res_dict, res_str

    def longtail_part_stagewise_eval(evaluator, mode):
        longtail_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        res_dict = {}
        res_str = "\nStagewise longtail part recall:\n"
        for hit_type, stat in evaluator.relation_per_cls_hit_recall.items():
            print(stat.shape)
            res_str += f"{hit_type}:\n"
            for topk_idx in range(len(stat)):
                each_stat = stat[topk_idx]
                recall_score = (each_stat[:, 0] / (each_stat[:, 1] + 1e-5))[1:].tolist()
                part_recall = {"h": [], "b": [], "t": [], }
                for idx, each_cat_recall in enumerate(recall_score):
                    part_recall[longtail_part_dict[idx + 1]].append(each_cat_recall)

                res_dict[f"sgdet_stagewise_longtail_part_recall/{hit_type}/top{topk_range[topk_idx]}/head"] = np.mean(part_recall['h'])
                res_dict[f"sgdet_stagewise_longtail_part_recall/{hit_type}/top{topk_range[topk_idx]}/body"] = np.mean(part_recall['b'])
                res_dict[f"sgdet_stagewise_longtail_part_recall/{hit_type}/top{topk_range[topk_idx]}/tail"] = np.mean(part_recall['t'])
                res_str += f"head: {np.mean(part_recall['h']):.4f} " \
                        f"body: {np.mean(part_recall['b']):.4f} " \
                        f"tail: {np.mean(part_recall['t']):.4f} @top{topk_range[topk_idx]}\n"
        res_str += '\n'
        return res_dict, res_str

    def unseen_eval(evaluator, zs_predicate, mode):
        assert "mean_recall" in evaluator.type
        unseen_marker = [False for _ in range(len(predicates_categories))]
        for zs_id in zs_predicate:
            unseen_marker[zs_id] = True

        res_dict = {}
        res_str = "\nOpenworld recall:\n"
        for topk, cate_rec_list in evaluator.result_dict[f'{mode}_{evaluator.type}_list'].items():
            part_recall = {"seen": [], "unseen": [] }
            for idx, each_cat_recall in enumerate(cate_rec_list):
                if unseen_marker[idx]:
                    part_recall['unseen'].append(each_cat_recall)
                else:
                    part_recall['seen'].append(each_cat_recall)
            res_dict[f"sgdet_seen_recall/top{topk}/seen"] = np.mean(part_recall['seen'])
            res_dict[f"sgdet_unseen_recall/top{topk}/unseen"] = np.mean(part_recall['unseen'])
            
            res_str += f"Top{topk:4}: unseen: {np.mean(part_recall['unseen']):.4f} " \
                       f"seen: {np.mean(part_recall['seen']):.4f} \n"

        return res_dict, res_str

    def unseen_stagewise_eval(evaluator, zs_predicate, mode):
        unseen_marker = [False for _ in range(len(predicates_categories))]
        for zs_id in zs_predicate:
            unseen_marker[zs_id] = True
        
        res_dict = {}
        res_str = "\nStagewise Openworld part recall:\n"
        for hit_type, stat in evaluator.relation_per_cls_hit_recall.items():
            stat = stat[-1]
            recall_score = (stat[:, 0] / (stat[:, 1] + 1e-5))[1:].tolist()
            part_recall = {"seen": [], "unseen": [] }
            for idx, each_cat_recall in enumerate(recall_score):
                if unseen_marker[idx + 1]:
                    part_recall['unseen'].append(each_cat_recall)
                else:
                    part_recall['seen'].append(each_cat_recall)

            res_dict[f"sgdet_stagewise_openworld_recall/{hit_type}/top100/seen"] = np.mean(part_recall['seen'])
            res_dict[f"sgdet_stagewise_openworld_part_recall/{hit_type}/top100/unseen"] = np.mean(part_recall['unseen'])
            res_str += f"{hit_type}: seen: {np.mean(part_recall['seen']):.4f} " \
                       f"unseen: {np.mean(part_recall['unseen']):.4f}\n"
        res_str += '\n'
        return res_dict, res_str

    # longtail_part_res_dict, longtail_part_res_str = longtail_part_eval(eval_mean_recall, mode)
    # ng_longtail_part_res_dict, ng_longtail_part_res_str = longtail_part_eval(eval_ng_mean_recall, mode)
    # stgw_longtail_part_res_dict, stgw_longtail_part_res_str = longtail_part_stagewise_eval(eval_stagewise_recall, mode)


    # print result
    result_str += eval_recall.generate_print_string(mode)
    result_str += eval_nog_recall.generate_print_string(mode)
    result_str += eval_zeroshot_recall.generate_print_string(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    if zeroshot_predicate is not None:
        result_str += eval_mean_recall.eval_given_cate_set_perf(mode, zeroshot_predicate)
        unseen_res_dict, unseen_res_str = unseen_eval(eval_mean_recall, zeroshot_predicate, mode)
        result_str += unseen_res_str
        stg_unseen_res_dict, stg_unseen_res_str = unseen_stagewise_eval(eval_stagewise_recall, zeroshot_predicate, mode)
        result_str += stg_unseen_res_str
        result_dict_list_to_log.append(stg_unseen_res_dict)
    
    result_str += eval_ng_mean_recall.generate_print_string(mode)
    result_str += eval_stagewise_recall.generate_print_string(mode)

    result_str += eval_recall_phrdet.generate_print_string('phrdet')
    result_str += eval_mean_recall_phrdet.generate_print_string('phrdet')
    # result_str += longtail_part_res_str
    # result_str += stgw_longtail_part_res_str
    # result_str += f"(Non-Graph-Constraint) {ng_longtail_part_res_str}"


    # entity grouping performance
    indexing_acc_res_str = ""
    for k, v in indexing_acc.items():
        if len(v) > 0:
            v = np.array(v)
            indexing_acc_res_str += f'{k}: {np.mean(v):.3f}\n'
    if len(indexing_acc_res_str) > 0:
        result_str += "indexing module acc: \n" + indexing_acc_res_str

    result_dict_list_to_log.extend([generate_eval_res_dict(eval_recall, mode),
                                    generate_eval_res_dict(eval_nog_recall, mode),
                                    generate_eval_res_dict(eval_zeroshot_recall, mode),
                                    generate_eval_res_dict(eval_mean_recall, mode),
                                    generate_eval_res_dict(eval_ng_mean_recall, mode),
                                    eval_stagewise_recall.generate_res_dict(mode),
                                    # longtail_part_res_dict, ng_longtail_part_res_dict
                                    ])

    result_str += '=' * 100 + '\n'
    # average the all recall and mean recall with the weight
    avg_metrics = np.mean(rel_eval_result_dict[mode + '_recall'][100]) * 0.5 \
                  + np.mean(rel_eval_result_dict[mode + '_mean_recall'][100]) * 0.5

    logger.info(result_str)

    result_dict = {}
    for each in result_dict_list_to_log:
        result_dict.update(each)
    return float(avg_metrics), result_dict


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths': groundtruths, 'predictions': predictions},
                   os.path.join(output_folder, "eval_results.pytorch"))

        # with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # jupyter information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
            ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
            ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
            })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field(
        'rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field(
        'pred_rel_score').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    local_container['rel_dist'] = prediction.get_field(
        'pred_rel_dist').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    local_container['rel_cls'] = prediction.get_field(
        'pred_rel_label').detach().cpu().numpy()  # (#pred_rels, num_pred_class)
    if prediction.has_field('rel_vec'):
        local_container['rel_vec'] = prediction.get_field('rel_vec').detach().cpu().numpy()

        # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field(
        'pred_labels').long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls
    if mode != 'sgdet':
        if evaluator.get("eval_pair_accuracy") is not None:
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    if evaluator.get("eval_zeroshot_recall") is not None:
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    if evaluator.get("eval_nog_recall") is not None:
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    if evaluator.get("eval_pair_accuracy") is not None:
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    if evaluator.get("eval_mean_recall") is not None:
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)


    if evaluator.get("eval_recall_phrdet") is not None:
        evaluator['eval_recall_phrdet'].calculate_recall(global_container, local_container, mode='phrdet')

    if evaluator.get("eval_mean_recall_phrdet") is not None:
        evaluator['eval_mean_recall_phrdet'].collect_mean_recall_items(global_container, local_container, mode='phrdet')


    if evaluator.get("eval_ng_mean_recall") is not None:
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    if evaluator.get("eval_zeroshot_recall") is not None:
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # 
    if evaluator.get('eval_rel_vec_recall') is not None:
        evaluator['eval_rel_vec_recall'].calculate_recall(global_container, local_container, mode)

    # stage wise recall
    if evaluator.get("eval_stagewise_recall") is not None:
        evaluator['eval_stagewise_recall'] \
            .calculate_recall(mode, global_container,
                              gt_boxlist=groundtruth.convert('xyxy').to("cpu"),
                              gt_relations=groundtruth.get_field('relation_tuple').long().detach().cpu(),
                              pred_boxlist=prediction.convert('xyxy').to("cpu"),
                              pred_rel_pair_idx=prediction.get_field('rel_pair_idxs').long().detach().cpu(),
                              pred_rel_dist=prediction.get_field('pred_rel_dist').detach().cpu())
    return



ENTITY_LONGTAIL_DICT=['t', 't', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't', 't', 'b', 't',
                        't', 'b', 'b', 'h', 'b', 't', 't', 'b', 't', 'b', 't', 't', 't', 't', 't', 't', 't', 't',
                        't', 'b', 't', 'b', 't', 't', 'b', 'b', 'b', 't', 't', 'b', 'b', 't', 't', 't', 'b', 'b',
                        't', 't', 'b', 'b', 'b', 'b', 'b', 'b', 't', 'b', 't', 'b', 'b', 't', 't', 't', 't', 't',
                        'b', 'b', 'b', 'b', 't', 'h', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 't', 'b',
                        'h', 't', 'b', 't', 't', 'b', 'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 't', 't', 't',
                        't', 'b', 'h', 'b', 'b', 'b', 'b', 't', 't', 't', 't', 't', 'b', 't', 't', 'b', 't', 'b',
                        'b', 't', 'b', 'b', 't', 't', 't', 'b', 'b', 'h', 't', 'b', 'b', 't', 't', 't', 'b', 'b',
                        'h', 't', 't', 't', 'h', 't'],
ENTITY_SORTED_CLS_LIST=[77, 135, 144, 110, 90, 21, 148, 114, 73, 60, 98, 56, 57, 125, 25, 75, 89, 86, 111, 39,
                        37, 44, 72, 27, 53, 65, 96, 123, 2, 143, 120, 126, 43, 59, 113, 112, 103, 3, 47, 58, 19,
                        133, 104, 61, 138, 134, 83, 52, 20, 99, 16, 129, 66, 74, 95, 128, 142, 42, 48, 9, 137,
                        63, 92, 22, 109, 18, 10, 40, 51, 76, 82, 13, 29, 17, 36, 80, 64, 136, 94, 146, 107, 79,
                        32, 87, 54, 149, 147, 30, 12, 14, 24, 4, 62, 97, 33, 116, 31, 70, 117, 124, 81, 23, 11,
                        26, 6, 108, 93, 145, 68, 121, 7, 84, 8, 46, 71, 28, 34, 15, 141, 102, 45, 131, 115, 41,
                        127, 132, 101, 88, 91, 122, 139, 5, 49, 100, 1, 85, 35, 119, 106, 38, 118, 105, 69, 130,
                        50, 78, 55, 140, 67]

def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """
    from list of attribute indexs to [1,0,1,0,...,0,1] form
    """
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets



def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def create_table_with_header(header_dict, headers=["category", "AP"], min_cols=6):
    """
    create a table with given header.

    Args:
        header_dict (dict):
        headers (list):
        min_cols (int):

    Returns:
        str: the table as a string
    """
    assert min_cols % len(headers) == 0, "bad table format"
    num_cols = min(min_cols, len(header_dict) * len(headers))
    result_pair = [x for pair in header_dict.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f",
        headers=headers * (num_cols // len(headers)),
        numalign="left")
    return table
