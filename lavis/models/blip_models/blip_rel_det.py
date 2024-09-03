"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from pprint import pprint
import copy
import json
import pickle
import os
import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from lavis.common.dist_utils import get_world_size
import numpy as np

from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.datasets.datasets.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from lavis.models.blip_models.blip_det import BlipDetection, StageBertTokenizer, check_label_name, find_token
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import SPEICAL_TOKEN_NUM, XBertLMHeadDecoder
from lavis.models.resnet import BasicBlock, BasicStem, Res18Wrapper
from lavis.models.vit import VisionTransformerEncoder
from lavis.models.detr_transformer import build_decoder, build_encoder, PositionEmbeddingSine, MLP
from lavis.tasks.evaluation.boxlist import BoxList

SPEC = -2
WORD = -3
COORD = -4
NOTIF = -5

logger = logging.getLogger(__name__)


@registry.register_model("blip_rel_detection")
class BlipRelDetection(BlipDetection):
    """
    BLIP captioning model.

    Supported model types:
        - base_coco: fine-tuned BLIP base model on COCO caption dataset (Karparthy split).
        - large_coco: fine-tuned BLIP large model on COCO caption dataset (Karparthy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_caption", "base_coco")
        >>> model = load_model("blip_caption", "large_coco")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_oiv6": "configs/models/blip_vrd_base_oiv6.yaml",
        "base_gqa": "configs/models/blip_vrd_base_gqa.yaml",
        "base_vg": "configs/models/blip_vrd_base_vg.yaml",
        "base_psg": "configs/models/blip_vrd_base_psg.yaml",
    }

    def __init__(self, image_encoder, text_decoder, prompt=None, max_txt_len=40,
                 max_objects=99, max_pos_objects=20, num_coord_bin=1000, add_noise=False,
                 dump_pred=False, reduction='none', top_k_label_num=5, top_k_predicate_label_num=3,
                 mask_label_ratio=0.5, aux_close_classifier=False, cate_dict_url="", box_loss_weight=1.,
                 dump_dir='/mnt/petrelfs/lirongjie/project/LAVIS/lavis/output/BLIP/vis_dump'):
        super().__init__(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len,
                         num_coord_bin=num_coord_bin, reduction=reduction, cate_dict_url=cate_dict_url)
        self.max_objects = max_objects  # all object for training
        self.max_pos_objects = max_pos_objects  # the number of postive sample
        self.add_noise = add_noise
        self.dump_pred = dump_pred
        self.dump_dir = dump_dir
        self.num_coord_bin = num_coord_bin

        self.top_k_label_num = top_k_label_num
        self.top_k_predicate_label_num = top_k_predicate_label_num
        self.mask_label_ratio = mask_label_ratio
        self.box_loss_weight = box_loss_weight

        with open(cate_dict_url, 'r') as f:
            self.cate_dict = json.load(f)

        self.aux_close_classifier = aux_close_classifier

        self.init_category_space()

    def init_category_space(self):

        self.rel_label_token = self.tokenizer(
            self.cate_dict['rel'],
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).input_ids[:, 1:]

        self.ent_label_token = self.tokenizer(
            self.cate_dict['obj'],
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).input_ids[:, 1:]

        if self.aux_close_classifier:
            # self.ent_close_classifier = nn.Sequential(
            #     nn.Linear(768, 768),
            #     nn.GELU(),
            #     nn.LayerNorm(768),
            #     nn.Linear(768, len(self.cate_dict['obj'])),
            # )

            # self.predicate_close_classifier = nn.Sequential(
            #     nn.Linear(768, 768),
            #     nn.GELU(),
            #     nn.LayerNorm(768),
            #     nn.Linear(768, len(self.cate_dict['rel'])),
            # )

            self.close_classifier = nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Linear(
                    768, len(self.cate_dict['obj']) + len(self.cate_dict['rel']), bias=False),
            )

    def get_classifier_weight(self, label_type):
        # extract classifier weights
        all_vocab_weight = copy.deepcopy(
            self.text_decoder.cls.predictions.decoder.weight.data)
        if label_type == 'rel':
            tgt_label_token = self.rel_label_token
        elif label_type == 'ent':
            tgt_label_token = self.ent_label_token

        tgt_label_token[tgt_label_token == 102] = 0
        label_weight = []
        token_range = []
        for each_cate in tgt_label_token:
            label_weight.append(
                all_vocab_weight[each_cate[each_cate.nonzero().view(-1)]])
            token_range.append(len(each_cate.nonzero().view(-1)))
        all_label_cls_weight = torch.cat(label_weight, dim=0)

        return all_label_cls_weight, token_range

    def get_classifier_scores(self, pred_prob, label_type):
        # extract classifier scores
        # pred_prob
        if label_type == 'rel':
            tgt_label_token = self.rel_label_token
        elif label_type == 'ent':
            tgt_label_token = self.ent_label_token

        tgt_label_token[tgt_label_token == 102] = 0
        pred_scores_filtered = []
        token_range = []
        for each_cate in tgt_label_token:
            pred_scores_filtered.append(
                pred_prob[:, each_cate[each_cate.nonzero().view(-1)]])
            token_range.append(len(each_cate.nonzero().view(-1)))
        all_label_score_filtered = torch.cat(pred_scores_filtered, dim=1)

        return all_label_score_filtered, token_range

    def vocab2category(self, pred_instances):

        ent_clser_weight, ent_cls_token_range = self.get_classifier_weight(
            'ent')
        rel_clser_weight, pred_cls_token_range = self.get_classifier_weight(
            'rel')

        device = rel_clser_weight.device
        for bi, b_pred_inst in enumerate(pred_instances):
            obj_ent_hs = []
            sub_ent_hs = []
            predicate_hs = []
            obj_pred_token_range = []
            sub_pred_token_range = []
            predicate_token_range = []

            for each_inst in b_pred_inst:
                obj_ent_hs.append(each_inst['obj']['dec_hs'])
                sub_ent_hs.append(each_inst['sub']['dec_hs'])
                predicate_hs.append(each_inst['predicate']['dec_hs'])
                obj_pred_token_range.append(len(each_inst['obj']['dec_hs']))
                sub_pred_token_range.append(len(each_inst['sub']['dec_hs']))
                predicate_token_range.append(
                    len(each_inst['predicate']['dec_hs']))

            assert len(obj_ent_hs) == len(b_pred_inst)

            if len(obj_ent_hs) == 0:
                continue

            all_obj_pred_hs = torch.cat(obj_ent_hs, dim=0)
            all_sub_pred_hs = torch.cat(sub_ent_hs, dim=0)
            all_pred_pred_hs = torch.cat(predicate_hs, dim=0)

            all_obj_pred_scores = self.text_decoder.cls.predictions(
                all_obj_pred_hs)
            all_sub_pred_scores = self.text_decoder.cls.predictions(
                all_sub_pred_hs)
            all_pred_pred_scores = self.text_decoder.cls.predictions(
                all_pred_pred_hs)

            all_obj_pred_scores = self.text_decoder.cls.predictions(
                all_obj_pred_hs).softmax(dim=-1)
            all_sub_pred_scores = self.text_decoder.cls.predictions(
                all_sub_pred_hs).softmax(dim=-1)
            all_pred_pred_scores = self.text_decoder.cls.predictions(
                all_pred_pred_hs).softmax(dim=-1)

            # all_obj_pred_scores = self.text_decoder.cls.predictions(
            #     all_obj_pred_hs).sigmoid()
            # all_sub_pred_scores = self.text_decoder.cls.predictions(
            #     all_sub_pred_hs).sigmoid()
            # all_pred_pred_scores = self.text_decoder.cls.predictions(
            #     all_pred_pred_hs).sigmoid()

            obj_pred_scores, _ = self.get_classifier_scores(
                all_obj_pred_scores, 'ent')
            sub_pred_scores, _ = self.get_classifier_scores(
                all_sub_pred_scores, 'ent')
            pred_pred_scores, _ = self.get_classifier_scores(
                all_pred_pred_scores, 'rel')

            # obj_pred_scores = obj_pred_scores.softmax(dim=-1)
            # sub_pred_scores = sub_pred_scores.softmax(dim=-1)
            # pred_pred_scores = pred_pred_scores.softmax(dim=-1)
            # obj_cos_sim = all_obj_pred_hs @ ent_clser_weight.T # num_perd_token, all_cate_pred_token
            # sub_cos_sim = all_sub_pred_hs @ ent_clser_weight.T
            # predi_cos_sim = all_pred_pred_hs @ rel_clser_weight.T

            def range_sum(sim_mat, range_ten):
                # sim_mat = torch.softmax(sim_mat, dim=-1) # num_inst, cat_all
                all_sim_sum = []
                # num_inst, cat_size
                for each_seg in sim_mat.split(range_ten, dim=-1):
                    # print(each_seg)
                    all_sim_sum.append(torch.mean(
                        each_seg, dim=-1))  # 1, num_inst
                    # all_sim_sum.append(torch.max(each_seg, dim=-1)[1]) # 1, num_inst
                return torch.stack(all_sim_sum).T  # num_inst, cate_size

            scores_all = {
                #  vocab2cla,
                'obj': range_sum(obj_pred_scores, ent_cls_token_range),
                'sub': range_sum(sub_pred_scores, ent_cls_token_range),
                'predicate': range_sum(pred_pred_scores, pred_cls_token_range)
            }

            token_range_all = {
                #  vocab2cla,
                'obj': obj_pred_token_range,
                'sub': sub_pred_token_range,
                'predicate': predicate_token_range
            }
            # token2inst
            for fld_name in ['obj', 'sub', 'predicate']:
                scores_all[fld_name] = range_sum(
                    scores_all[fld_name].T, token_range_all[fld_name]).T

            if self.aux_close_classifier:
                dec_hs = {
                    'obj': obj_ent_hs,
                    'sub': sub_ent_hs,
                    'predicate': predicate_hs,
                }
                scores_all_close = {}
                for fld_name in ['obj', 'sub', 'predicate']:
                    avg_hs = []
                    for inst in dec_hs[fld_name]:
                        # avg_hs.append(inst[0].mean(0))
                        avg_hs.append(inst[0])
                    avg_hs = torch.stack(avg_hs)
                    scores = self.close_classifier(avg_hs)
                    if fld_name != 'predicate':
                        scores = scores[:, :len(
                            self.cate_dict['obj'])].softmax(-1)
                    else:
                        scores = scores[:, len(
                            self.cate_dict['obj']):].softmax(-1)
                    scores_all_close[fld_name] = scores

                scores_all = scores_all_close

            num_cate = 30
            for inst_id in range(len(scores_all['obj'])):
                for fld_name in ['obj', 'sub', 'predicate']:
                    pred_instances[bi][inst_id][fld_name]['scores_mapped'] = scores_all[fld_name][inst_id]
                    # print(pred_instances[bi][inst_id][fld_name]['labels_mapped'])
                    # pred_instances[bi][inst_id][fld_name]['labels_mapped'] = torch.multinomial(
                    #     torch.softmax(scores_all[fld_name][inst_id], dim=-1), num_samples=5)
                    # print(pred_instances[bi][inst_id][fld_name]['labels_mapped'])
                    # pred_instances[bi][inst_id][fld_name]['labels_mapped'] = torch.multinomial(scores_all[fld_name][inst_id],
                    #                                                                            num_samples=num_cate)

                    pred_instances[bi][inst_id][fld_name]['labels_mapped'] = torch.topk(scores_all[fld_name][inst_id],
                                                                                        k=num_cate, dim=-1)[1]
                    # print(pred_instances[bi][inst_id][fld_name]['labels_mapped'])

        return pred_instances

    @classmethod
    def init_tokenizer(cls):
        tokenizer = StageBertTokenizer.from_pretrained("bert-base-uncased",)
        vocab_size = len(tokenizer)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

        tokenizer.add_tokens(["[CORD]"])
        tokenizer.obj_coord_token_id = len(tokenizer) - 1
        tokenizer.add_tokens(["[OBJ]"])
        tokenizer.obj_s_token_id = len(tokenizer) - 1
        tokenizer.add_tokens(["[OBJ/]"])
        tokenizer.obj_e_token_id = len(tokenizer) - 1

        tokenizer.add_tokens(["[NOS]"])
        tokenizer.noise_token_id = len(tokenizer) - 1

        tokenizer.add_tokens(["[REL]"])
        tokenizer.obj_rel_token_id = len(tokenizer) - 1

        special_token_num = len(tokenizer) - vocab_size
        logger.info(f"special_token_num: {special_token_num}")

        return tokenizer, vocab_size, special_token_num

    def forward_encoder(self, samples):
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        return image_embeds

    def forward_decoder(self, samples, image_embeds, return_close_vocab_classes=False):
        # prepare inputs for forwarding decoder
        targets = samples["targets"]

        label_token, decoder_targets, close_vocab_classes, box_gts_batch = self.target2token_seqs(
            targets, text_mask_ratio=0.1)
        # close_vocab_classes: batch_size, instance_size, tuple(sub, predicate obj)

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        model_config = {
            'vocab_size': len(self.tokenizer),
            'coord_token_start': self.coord_token_start,
            'text_vocab_size': self.text_vocab_size,
            'num_coord_bins': self.num_coord_bins
        }

        decoder_output = self.text_decoder(
            input_ids=label_token.input_ids,
            attention_mask=label_token.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            box_targets=box_gts_batch,
            reduction=self.loss_reduction,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            model_config=model_config,
            box_loss_weights_all=self.box_loss_weight,
            raw_image=samples["image"],
        )

        if self.aux_close_classifier:
            # odict_keys(['loss', 'logits', 'hidden_states'])
            batch_text_token = self.seq2instances_train(decoder_targets)
            decoder_output = self.vocab2category_train(
                batch_text_token, decoder_output, close_vocab_classes, decoder_targets)
        if return_close_vocab_classes:
            return decoder_output, decoder_targets, close_vocab_classes
        return decoder_output, decoder_targets

    def seq2instances_train(self, decoder_targets):

        def get_token_type(labels):
            token_types = torch.zeros_like(labels)
            token_types[torch.logical_and(torch.logical_and(labels >= 105, labels < 30524),
                                          labels != 1010)] = WORD
            # token_types[torch.logical_and(labels >= 103, labels < 30524)] = WORD
            token_types[torch.logical_and(
                labels >= 30524, labels < 30534)] = NOTIF
            token_types[labels >= 30534] = COORD
            return token_types
        # remove the first token targets to align with hidden state
        decoder_targets = decoder_targets[:, 1:].contiguous()
        token_type = get_token_type(decoder_targets)
        text_idx = token_type == WORD
        batch_text_token = []
        for b_id, text_id_bi in enumerate(text_idx):
            text_tok_idx = (text_id_bi).nonzero().view(-1)
            trp_token_idx = [[], [], []]
            role_id = 0
            # instance_len, 3 (subject, predicate, object)
            all_trp_token_idx = []
            for i, text_id in enumerate(text_tok_idx):
                trp_token_idx[role_id].append(text_id.item())

                if i+1 >= len(text_tok_idx):  # last token return
                    all_trp_token_idx.append(trp_token_idx)
                    break

                if text_tok_idx[i+1] - text_id > 1:  # next role
                    role_id += 1

                if role_id > 2:
                    all_trp_token_idx.append(trp_token_idx)
                    trp_token_idx = [[], [], []]
                    role_id = 0

            batch_text_token.append(all_trp_token_idx)

        return batch_text_token

    def vocab2category_train(self, batch_text_token, decoder_output, close_vocab_classes, decoder_targets):
        decoder_hs = decoder_output.last_hidden_states  # b, seq_len, dim
        all_label_hs = []
        all_label_cate = []
        batch_inst_len = []
        all_role_marker = []
        for b_id, text_id_bi in enumerate(batch_text_token):
            decoder_hs_b = decoder_hs[b_id]
            label_id = []
            label_hs = []
            role_mark = []
            PRED = 0
            ENT = 1
            for inst_i, each_inst_token_idxs in enumerate(text_id_bi):
                for role_i, role_token_idx in enumerate(each_inst_token_idxs):
                    try:
                        if role_i == 1:
                            role_mark.append(PRED)
                            label_id.append(
                                close_vocab_classes[b_id][inst_i][role_i] + len(self.cate_dict['obj']))
                        else:
                            role_mark.append(ENT)
                            label_id.append(
                                close_vocab_classes[b_id][inst_i][role_i])
                        # pred_labels = self.text_decoder.cls.predictions(decoder_hs_b[role_token_idx]).max(-1)[1]
                        # gt_lables = decoder_targets[b_id][role_token_idx]
                        # print(pred_labels, gt_lables)

                        # torch.mean(decoder_hs_b[role_idx], dim=0)
                        label_hs.append(decoder_hs_b[role_token_idx][0])
                    except:
                        print(close_vocab_classes)
                        # todo some categories has "and" word

            all_role_marker.append(torch.Tensor(role_mark).long())
            all_label_hs.append(torch.stack(label_hs))
            all_label_cate.append(torch.Tensor(label_id).long())
            batch_inst_len.append(len(text_id_bi))

        # classifier
        all_label_hs = torch.cat(all_label_hs, dim=0)
        all_label_cate = torch.cat(all_label_cate).to(self.device)
        all_label_logits = self.close_classifier(all_label_hs)

        # calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none", label_smoothing=0.003, ignore_index=-100)
        loss_close_set = loss_fct(
            all_label_logits, all_label_cate).mean() * np.mean(batch_inst_len)

        decoder_output.loss['loss_close_set'] = loss_close_set
        decoder_output.loss['loss'] += loss_close_set

        word_acc = sum(all_label_logits.max(-1)
                       [1] == all_label_cate) / all_label_cate.view(-1).shape[0]
        decoder_output.loss['close_vocab_acc'] = word_acc

        return decoder_output

    def target2token_seqs(self, targets, text_mask_ratio=0.5):

        add_noise = self.add_noise
        tar_seqs = []
        # construct the templets
        batch_num_noise = []
        for b_i, target in enumerate(targets):
            # dict_keys(['boxes', 'det_labels', 'rel_tripets', 'image_id', 'orig_size', 'size', 'det_labels_texts', 'rel_labels_texts'])
            all_idx = torch.randperm(len(target['rel_tripets'])).tolist()
            if add_noise:
                if len(all_idx) > self.max_pos_objects:
                    all_idx = all_idx[:self.max_pos_objects]
            else:
                if len(all_idx) > self.max_objects:
                    all_idx = all_idx[:self.max_objects]

            target['rel_tripets'] = target['rel_tripets'][all_idx]
            det_label = target['det_labels']
            rel_tri_text = []
            for each_trp in target['rel_tripets']:
                trp_idx = each_trp.tolist()
                rel_tri_text.append(
                    f"{self.cate_dict['obj'][det_label[trp_idx[0]]]} [OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/] {self.cate_dict['rel'][trp_idx[2]]} [REL] {self.cate_dict['obj'][det_label[trp_idx[1]]]} [OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/]")

            seperate_word = " and "
            # print(rel_tri_text)
            # seperate_word = " , "
            tar_seq = self.prompt + seperate_word.join(rel_tri_text)

            num_noise = 0
            if add_noise:
                num_pos = len(target['rel_tripets'])
                nos_rate = 2
                num_noise = num_pos * \
                    nos_rate if self.max_objects > (
                        1 + nos_rate) * num_pos else self.max_objects - num_pos
                tar_seq += '[SEP]'
            batch_num_noise.append(num_noise)

            noise_targets = []
            random_class_obj = torch.randint(
                0, len(self.cate_dict['obj']), (num_noise, 2)).numpy()
            random_class_pred = torch.randint(
                0, len(self.cate_dict['rel']), (num_noise, )).numpy()

            for i in range(num_noise):
                inst = f"{self.cate_dict['obj'][random_class_obj[i, 0]]} [OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/] {self.cate_dict['rel'][random_class_pred[i]]} [REL] {self.cate_dict['obj'][random_class_obj[i, 1]]} [OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/] [SEP]"
                noise_targets.append(inst)
            tar_seq += seperate_word.join(noise_targets)

            tar_seqs.append(tar_seq)

        label_token = self.tokenizer(
            tar_seqs,
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        # label_token.input_ids[label_token.input_ids == 102] = 0
        label_token.input_ids[:, 0] = self.tokenizer.bos_token_id
        # print(label_token.input_ids.shape)
        raw_token = []
        close_vocab_classes = []
        box_gts_batch = []
        for b_i, target in enumerate(targets):
            boxes = target['boxes']
            boxes_xywh = copy.deepcopy(boxes)
            box_tokens = (boxes_xywh * self.num_coord_bins).floor().long().clamp(min=0,
                                                                                 max=self.num_coord_bins) + self.coord_token_start
            coord_id = 0
            rel_idx = 0

            label_token_fuse = label_token['input_ids'][b_i, :]
            close_vocab_cate_id = []
            box_gt = []
            for trp_idx, each_trp in enumerate(target['rel_tripets']):
                coord_id = 0
                close_vocab_cate_id.append((
                    target['det_labels'][each_trp[0]].item(),
                    each_trp[-1], target['det_labels'][each_trp[1]].item()
                ))

                while coord_id < 8 and rel_idx + 5 < len(label_token_fuse):

                    if label_token_fuse[rel_idx] == self.tokenizer.obj_s_token_id:
                        if coord_id < 4:
                            curr_box_coord = box_tokens[each_trp[0]]
                            box_gt.append({
                                'xywh': boxes_xywh[each_trp[0]],
                                'tkn_pos': (rel_idx + 1,  rel_idx + 5)
                            })
                        else:
                            curr_box_coord = box_tokens[each_trp[1]]
                            box_gt.append({
                                'xywh': boxes_xywh[each_trp[1]],
                                'tkn_pos': (rel_idx + 1,  rel_idx + 5)
                            })

                        label_token_fuse[rel_idx +
                                         1: rel_idx + 5] = curr_box_coord

                        coord_id += 4
                        rel_idx += 5

                        # print("rel_idx", rel_idx, "trp_idx", trp_idx, "coord_id", coord_id)
                        # print(label_token_fuse)
                        continue
                    rel_idx += 1

            box_gts_batch.append(box_gt)

            if add_noise:
                num_pos = len(target['rel_tripets'])
                num_noise = batch_num_noise[b_i]
                num_noise *= 2  # subject and object boxes

                random_box_x0y0 = torch.rand(num_noise, 2)
                random_box_wh = torch.rand(num_noise, 2)
                random_box_x1y1 = (random_box_x0y0 +
                                   random_box_wh).clamp(min=0, max=1)
                random_box = torch.cat(
                    [random_box_x0y0, random_box_x1y1], dim=1)
                random_box_tokens = (random_box * self.num_coord_bins).floor().long().clamp(
                    min=0, max=self.num_coord_bins) + self.coord_token_start

                for n_obj_idx in range(num_noise):
                    while rel_idx < len(label_token_fuse):
                        if label_token_fuse[rel_idx] == self.tokenizer.obj_s_token_id:
                            curr_box_coord = random_box_tokens[n_obj_idx]
                            label_token_fuse[rel_idx +
                                             1: rel_idx + 5] = curr_box_coord
                            rel_idx += 4
                            break

                        rel_idx += 1

            raw_token.append(label_token_fuse)
            close_vocab_classes.append(close_vocab_cate_id)

        raw_token = torch.stack(raw_token)
        label_token.input_ids = raw_token

        # prepare targets for forwarding decoder
        decoder_targets = label_token.input_ids.masked_fill(
            label_token.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        if text_mask_ratio > 0:
            for b_i, target in enumerate(targets):
                # seletively mask label categories
                label_token_seq = label_token.input_ids[b_i]
                label_text_idx = torch.logical_and(
                    label_token_seq > 105, label_token_seq < self.text_vocab_size)[self.prompt_length:]
                label_text_idx = label_text_idx.nonzero().view(-1) + self.prompt_length
                masked_idx = label_text_idx[torch.randperm(len(label_text_idx))[
                    :int(len(label_text_idx) * text_mask_ratio)]]
                label_token_seq[masked_idx] = self.tokenizer.mask_token_id
                label_token.input_ids[b_i] = label_token_seq

            # mask noise instance labels and ignore noise coordination
            if add_noise:
                decoder_target_seq = decoder_targets[b_i]
                noise_token_start = (
                    decoder_targets[b_i] == self.tokenizer.sep_token_id).nonzero().view(-1)[0]
                rel_idx = noise_token_start.item()

                decoder_target_seq_noise_range = decoder_target_seq[rel_idx:]
                label_text_idx = torch.logical_and(
                    decoder_target_seq_noise_range > 105, decoder_target_seq_noise_range < self.text_vocab_size).nonzero().view(-1)
                decoder_target_seq_noise_range[label_text_idx] = self.tokenizer.noise_token_id
                decoder_target_seq[rel_idx:] = decoder_target_seq_noise_range

                while rel_idx < len(decoder_target_seq):
                    if decoder_target_seq[rel_idx] == self.tokenizer.obj_s_token_id:
                        decoder_target_seq[rel_idx + 1: rel_idx + 5] = -100
                        rel_idx += 4
                    rel_idx += 1

                decoder_targets[b_i] = decoder_target_seq

        return label_token, decoder_targets, close_vocab_classes, box_gts_batch

    def forward(self, samples):
        r"""
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size.
        Returns:
            output (BlipOutput): A BlipOutput object containing the following
                attributes:
                - loss (torch.Tensor): A scalar tensor containing the total loss. For BlipCaption, this is the same as the LM loss.
                - loss_lm (torch.Tensor): A scalar tensor containing the LM loss.
                - intermediate_outputs (BlipIntermediateOutput): A BlipIntermediateOutput object containing intermediate outputs.
                  see :class:`lavis.models.blip_models.blip_outputs.BlipOutput` for more details.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> text_input = ["a large statue of a person spraying water from a fountain"]
        >>> samples = {"image": image, "text_input": text_input}
        >>> output = model(samples)
        >>> output.keys()
        odict_keys(['intermediate_output', 'loss', 'loss_lm'])
        >>> output.intermediate_output.image_embeds.shape
        torch.Size([1, 577, 768])
        >>> output.intermediate_output.decoder_labels.shape
        torch.Size([1, 13])
        ```"""

        image_embeds = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(
            samples, image_embeds)

        # return decoder_out
        return BlipOutput(
            loss=decoder_output.loss,
            loss_lm=decoder_output.loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

    def sgg_cls(self, samples):
        image_embeds = self.forward_encoder(samples)

        targets = samples["targets"]

        label_token, decoder_targets, close_vocab_classes, box_gts_batch = self.target2token_seqs(
            targets, text_mask_ratio=1.0)
        # mask all instance for prediction

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        model_config = {
            'vocab_size': len(self.tokenizer),
            'coord_token_start': self.coord_token_start,
            'text_vocab_size': self.text_vocab_size,
            'num_coord_bins': self.num_coord_bins
        }

        decoder_output = self.text_decoder(
            input_ids=label_token.input_ids,
            attention_mask=label_token.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            box_targets=box_gts_batch,
            model_config=model_config,
            # labels=None,
            reduction=self.loss_reduction,
            output_hidden_states=True,
            return_dict=True,
            box_loss_weights_all=0.001,
        )

        # self.tokenizer.decode(label_token.input_ids[0].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
        pred_logits = decoder_output.logits
        # pred_logits[:, :, 102] = -1e10
        pred_scores = pred_logits.softmax(dim=-1)

        pred_index_sampled = []
        for each_b in pred_scores:
            pred_index_sampled.append(torch.multinomial(each_b, num_samples=1))
        pred_index_sampled = torch.stack(pred_index_sampled)[:, :, 0]
        pred_index = pred_index_sampled

        pred_index = pred_scores.max(dim=-1)[1]
        pred_index_filled = copy.deepcopy(pred_index)[:, :-1]
        decoder_targets_shifted = decoder_targets[:, 1:]

        boxes_coord_token_idx = torch.logical_or(decoder_targets_shifted >= (self.text_vocab_size),
                                                 decoder_targets_shifted == 1998)  # the 'and token'

        pred_index_filled[boxes_coord_token_idx] = decoder_targets_shifted[boxes_coord_token_idx]

        raw_caption = [self.tokenizer.decode(pred_index_filled[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                       for i in range(len(pred_index_filled))]

        prompt_length = 3
        decoder_hs = [each.unsqueeze(dim=-1)
                      for each in decoder_output.hidden_states]
        decoder_hs = torch.stack(decoder_hs).permute(2, 0, 1, 4, 3)

        decoder_out_dict = {
            # the prompt of sequences are cropped in seq2instance
            'sequences': pred_index_filled,
            # crop the prompt before for simulating the generation
            'decoder_hidden_states': decoder_hs[prompt_length:-1],
            'scores': pred_scores[:, prompt_length:-1, :].permute(1, 0, 2)
        }

        batch_object_list = self.seq2instance(
            decoder_out_dict, raw_caption, prompt_length=prompt_length)

        # for bi in range(len(batch_object_list)):
        #     print(len(batch_object_list[bi]), len(samples['targets'][bi]['rel_tripets']))

        # mapping prediction into fix label space
        batch_object_list_cate_trans = self.vocab2category(batch_object_list)

        predictions, ground_truths, image_info = self._postprocess(
            batch_object_list_cate_trans, samples['targets'])

        return predictions, ground_truths, image_info

    def generate(
        self,
        samples,
        use_nucleus_sampling=True,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                dict_keys(['image', 'targets', 'image_id', 'instance_id', 'image_pth'])
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.

        Example:
        ```python
        >>> from PIL import Image
        >>> from lavis.models import load_model_and_preprocess
        >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption")
        >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
        >>> samples = {"image": image}
        >>> captions = model.generate(samples)
        >>> captions
        ['a large statue of a person spraying water from a fountain']
        >>> captions = model.generate(samples, use_nucleus_sampling=True, num_captions=3)
        >>> captions # example output, results may vary due to randomness
        ['singapore showing the view of some building',
        'the singapore harbor in twilight, as the weather is going down',
        'the famous singapore fountain at sunset']
        """

        # prepare inputs for decoder generation.
        encoder_out = self.forward_encoder(samples)
        image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        # get decoded text
        # batch_target_object_list = None
        # raw_caption_target = None
        # decoder_targets = None

        # decoder_out_target = {
        #     'sequences': decoder_targets
        # }
        # raw_caption_target = [self.tokenizer.decode(decoder_targets[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
        #                         for i in range(len(decoder_targets))]
        # batch_target_object_list = self.seq2instance(decoder_out_target, raw_caption_target)

        # decoder_out_grdy = self.text_decoder.generate_from_encoder(
        #     tokenized_prompt=prompt,
        #     visual_embeds=image_embeds,
        #     sep_token_id=self.tokenizer.sep_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     use_nucleus_sampling=False,
        #     num_beams=num_beams,
        #     max_length=max_length,
        #     min_length=min_length,
        #     top_p=top_p,
        #     repetition_penalty=repetition_penalty,
        #     output_attentions=True,  # 16, 12, 192, 12, 4, 4
        #     return_dict_in_generate=True,
        #     output_hidden_states=True,
        #     output_scores=True,
        #     num_return_sequences=1,
        # )

        decoder_out = self.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            output_attentions=True,  # 16, 12, 192, 12, 4, 4
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            num_return_sequences=1,
        )

        # odict_keys(['sequences', 'decoder_attentions', 'cross_attentions', decoder_hidden_states])
        # decoder_out.cross_attentions:
        #       seq_len, decoder_layer, batch_size*num_captions, head_num, previous_seq_size(prompt_size or 1), image_hs_size
        # decoder_out.decoder_hidden_states:
        #       seq_len, decoder_layer, batch_size*num_beams, generate_len (第一个token因为是prompt大于1 其他都等于1), hid_dim
        raw_caption = [self.tokenizer.decode(decoder_out.sequences[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                       for i in range(len(decoder_out.sequences))]

        batch_object_list = self.seq2instance(decoder_out, raw_caption)
        # mapping prediction into fix label space
        #  (batch_object_list[0][i]['sub']['label_tkn'], self.text_decoder.cls.predictions(batch_object_list[0][i]['sub']['dec_hs']).topk(dim=-1, k=2)[1])

        # for i in range(11):
        #     for role in ['sub', 'predicate', 'obj']:
        #         (batch_object_list[0][i][role]['label_tkn'].tolist(), self.text_decoder.cls.predictions(batch_object_list[0][i][role]['dec_hs']).topk(dim=-1, k=2)[1].tolist(), self.close_classifier(batch_object_list[0][i][role]['dec_hs']).softmax(-1).topk(dim=-1, k=2))
        batch_object_list_cate_trans = self.vocab2category(batch_object_list)
        # for object_list_cate_trans in batch_object_list_cate_trans:
        #     for new_inst in object_list_cate_trans:
        #         for k,v in new_inst.items():
        #                 print(k)
        #                 vi = copy.copy(v)
        #                 vi.pop('dec_hs')
        #                 pprint(vi)
        if self.text_decoder.pos_adapter_on:
            pred_ent_hs_all = []
            # extract entity instance and categories hs
            encoder_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )
            for bid, adapter_box_pred in enumerate(batch_object_list_cate_trans):
                ent_hs = []
                for each_box_pred in adapter_box_pred:
                    for each_role in ['sub', 'obj']:
                        ent_hs.append(each_box_pred[each_role]['int_tok_hs'][-1] + each_box_pred[each_role]['dec_hs'][-1])
                pred_ent_hs_all.append(torch.stack(ent_hs))
            pos_ada_output = self.text_decoder.pos_adapter(
                pred_ent_hs_all, samples['image'], image_embeds, encoder_attention_mask, None)

            role_list = ['sub', 'obj']
            for bid, adapter_box_pred in enumerate(pos_ada_output['extracted_box']):
                for inst_id, each_box_pred in enumerate(adapter_box_pred):
                    init_token_boxes = batch_object_list_cate_trans[bid][int(inst_id//2)][role_list[inst_id % 2]]['boxes']
                    batch_object_list_cate_trans[bid][int(inst_id//2)][role_list[inst_id % 2]]['boxes'] = each_box_pred
                    batch_object_list_cate_trans[bid][int(inst_id//2)][role_list[inst_id % 2]]['token_boxes'] = init_token_boxes

        predictions, ground_truths, image_info = self._postprocess(
            batch_object_list_cate_trans, samples['targets'])

        if self.dump_pred:
            forward_decoder_output, decoder_targets, close_vocab_classes = self.forward_decoder(
                samples, image_embeds, return_close_vocab_classes=True)
            raw_caption_target = [self.tokenizer.decode(decoder_targets[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                  for i in range(len(decoder_targets))]
            f_dec_tokens = torch.argmax(
                forward_decoder_output.logits.contiguous(), dim=-1)
            raw_caption_fdec = [self.tokenizer.decode(f_dec_tokens[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                for i in range(len(f_dec_tokens))]

            # breakpoint()

            image_ids = samples['instance_id'][0]
            forward_buffer = {
                # 'cross_attentions': decoder_out['cross_attentions'],
                # 'decoder_attentions': decoder_out['decoder_attentions'],
                'scores': decoder_out['scores'],  # num_toke, bz, vocab_size
                # 'normed_image': samples['image'],
                'raw_token': decoder_out.sequences.detach().cpu(),
                "forward_decoder_output": forward_decoder_output,
                "f_dec_tokens": f_dec_tokens,

                'raw_caption_fdec': raw_caption_fdec,
                'raw_caption': raw_caption,
                "raw_caption_target": raw_caption_target,
                "decoder_targets": decoder_targets,
                # "batch_target_object_list": batch_target_object_list,

                "predictions": predictions,
                "ground_truths": ground_truths,

                'image_path': samples['image_pth'],
                "batch_object_list": batch_object_list,
                "gt_instance": samples['targets']
                # 'image_pixel': data_loader.dataset.__getitem__(img_idx, raw_image=True),
                # 'image_size': (image_size, patch_size)
            }

            if not os.path.exists(self.dump_dir):
                os.makedirs(self.dump_dir)
            logger.info(f"save data {image_ids} to {self.dump_dir}")

            with open(os.path.join(self.dump_dir, f'vl_det_dump_{image_ids}.pkl'), 'wb') as f:
                pickle.dump(forward_buffer, f)

            if image_ids > 48:
                exit()

        return predictions, ground_truths, image_info

    def seq2instance(self, decoder_out, raw_caption, prompt_length=None):
        """_summary_

            "{target['det_labels_texts'][trp_idx[0]]} [OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/] {target['rel_labels_texts'][idx]} [REL] {target['det_labels_texts'][trp_idx[1]]} [OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/]"

        Returns:
            _type_: _description_
        """
        batch_object_list = []

        if prompt_length is None:
            prompt_length = self.prompt_length

        all_inst_num = []
        vaild_inst_nun = []
        seq_len = []
        for bi, seq in enumerate(raw_caption):

            seq_no_pmpt = seq[prompt_length:]
            tok_seq_nppmpt = decoder_out['sequences'][bi][prompt_length:]
            seq_len.append(len(seq_no_pmpt))
            tok_hs_dump = None
            if decoder_out.get('decoder_hidden_states') is not None:
                tok_hs_dump = []
                for l_i, l in enumerate(decoder_out['decoder_hidden_states']):
                    tok_hs_dump.append(l[-1][bi][-1])
                tok_hs_dump = torch.stack(tok_hs_dump)  # remove prompt

            pred_score = None
            if decoder_out.get('scores') is not None:
                scores_tensor = []
                for each in decoder_out['scores']:
                    scores_tensor.append(each[bi].softmax(dim=-1))
                pred_score = torch.stack(scores_tensor)  # num_tok, vocab_size

            object_list = []
            last_start = 0
            curr_t_idx = last_start

            bad_inst = 0
            all_inst = 0

            # print()
            # print(seq_no_pmpt)
            while curr_t_idx < len(seq_no_pmpt):
                tok_word = seq_no_pmpt[curr_t_idx]
                if tok_word.upper() == '[REL]':
                    # print(last_start, curr_t_idx)
                    # print(seq_no_pmpt[last_start: curr_t_idx+8])
                    # print(tok_seq_nppmpt[last_start: curr_t_idx+8])

                    all_inst += 1
                    # find subject
                    last_obj_e = find_token(
                        seq_no_pmpt, '[obj/]', last_start, curr_t_idx)
                    last_obj_b = find_token(
                        seq_no_pmpt, '[obj]', last_start, curr_t_idx)

                    if len(last_obj_e) < 1 or len(last_obj_b) < 1:
                        # print('subject cropput')
                        bad_inst += 1
                        # update index
                        last_start = curr_t_idx + 1
                        curr_t_idx = last_start
                        continue

                    box_coord_b = last_obj_b[-1]
                    box_coord_e = last_obj_e[-1]
                    if box_coord_b > box_coord_e:
                        bad_inst += 1
                        last_start = curr_t_idx + 1
                        curr_t_idx = last_start
                        continue

                    if box_coord_e - box_coord_b != 5:
                        if box_coord_e - box_coord_b > 5:
                            # print('fixed as', obj_boxs[:4])
                            box_coord_e = box_coord_b + 5
                        else:
                            bad_inst += 1
                            last_start = curr_t_idx + 1
                            curr_t_idx += 1
                            continue

                    if len(tok_seq_nppmpt[box_coord_b + 1: box_coord_e]) != 4:
                        bad_inst += 1
                        last_start = curr_t_idx + 1
                        curr_t_idx += 1
                        continue

                    if len(last_obj_e) >= 2:
                        last_start = last_obj_e[-2] + 1

                    ent_cate_pred_start = last_start

                    if seq_no_pmpt[ent_cate_pred_start] == 'and' or seq_no_pmpt[ent_cate_pred_start] == ',':
                        ent_cate_pred_start += 1

                    ent_cate_pred_end = box_coord_b
                    ent_box_start = box_coord_b + 1
                    ent_box_end = box_coord_e

                    sub_ent_start = ent_cate_pred_start
                    sub_ent_end = ent_cate_pred_end

                    sub_label_name = seq_no_pmpt[ent_cate_pred_start: ent_cate_pred_end]
                    sub_label_token = tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]
                    if pred_score is not None:
                        sub_pred_scores = pred_score[ent_cate_pred_start: ent_cate_pred_end,
                                                     tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]]
                    else:
                        sub_pred_scores = torch.ones((1,))

                    sub_tok_hs = None
                    sub_int_tok_hs = None
                    if tok_hs_dump is not None:
                        sub_tok_hs = tok_hs_dump[ent_cate_pred_start: ent_cate_pred_end]
                        sub_int_tok_hs = tok_hs_dump[ent_cate_pred_end: ent_cate_pred_end+1]
                    else:
                        sub_tok_hs = torch.ones((768,)).to(self.device)
                        sub_int_tok_hs = torch.ones((768,)).to(self.device)

                    sub_boxs = tok_seq_nppmpt[ent_box_start: ent_box_end].detach(
                    ).cpu()
                    sub_boxs = (sub_boxs - self.coord_token_start) / \
                        self.num_coord_bins

                    if check_label_name(sub_label_name):
                        # print("corrput sub_label_name", sub_label_name)
                        bad_inst += 1
                        # update index
                        curr_t_idx = curr_t_idx + 1
                        last_start = curr_t_idx
                        continue

                    # get predicate prediction
                    pred_label_start = last_obj_e[-1] + 1
                    pred_label_end = curr_t_idx
                    predicate_label_name = seq_no_pmpt[pred_label_start: pred_label_end]
                    predicate_label_token = tok_seq_nppmpt[pred_label_start: pred_label_end]
                    if check_label_name(predicate_label_name):
                        # print("corrput predicate_label_name",predicate_label_name)
                        bad_inst += 1
                        curr_t_idx = curr_t_idx + 1
                        last_start = curr_t_idx
                        continue

                    predicate_pred_scores = None
                    if pred_score is not None:
                        predicate_pred_scores = pred_score[pred_label_start: pred_label_end,
                                                           tok_seq_nppmpt[pred_label_start: pred_label_end]]
                    else:
                        predicate_pred_scores = torch.ones((1,))

                    predicate_tok_hs = None
                    if tok_hs_dump is not None:
                        predicate_tok_hs = tok_hs_dump[pred_label_start: pred_label_end]
                    else:
                        predicate_tok_hs = torch.ones((768,)).to(self.device)

                    # get object
                    last_obj_b = find_token(
                        seq_no_pmpt, '[obj]', curr_t_idx, curr_t_idx+15)
                    last_obj_e = find_token(
                        seq_no_pmpt, '[obj/]', curr_t_idx, curr_t_idx+15)

                    if len(last_obj_e) < 1 or len(last_obj_b) < 1:
                        # print('object cropput')
                        # last_start = curr_t_idx + 1
                        bad_inst += 1
                        # last_start = last_obj_e[0] + 1
                        curr_t_idx = curr_t_idx + 1
                        last_start = curr_t_idx
                        continue

                    box_coord_b = last_obj_b[0]
                    box_coord_e = last_obj_e[0]

                    if box_coord_b > box_coord_e:
                        bad_inst += 1
                        last_start = curr_t_idx + 1
                        curr_t_idx += 1
                        continue

                    # adjust box length if it not match require
                    if box_coord_e - box_coord_b != 5:
                        if box_coord_e - box_coord_b > 5:
                            # print('fixed as', obj_boxs[:4])
                            box_coord_e = box_coord_b + 5
                        else:
                            bad_inst += 1
                            last_start = curr_t_idx + 1
                            curr_t_idx += 1
                            continue

                    # check box after adjust
                    if len(tok_seq_nppmpt[box_coord_b + 1: box_coord_e]) != 4:
                        bad_inst += 1
                        last_start = curr_t_idx + 1
                        curr_t_idx += 1
                        continue

                    ent_cate_pred_start = pred_label_end + 1
                    obj_ent_start = ent_cate_pred_start
                    obj_ent_end = ent_cate_pred_end

                    ent_cate_pred_end = box_coord_b
                    obj_label_name = seq_no_pmpt[ent_cate_pred_start: ent_cate_pred_end]
                    obj_label_token = tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]

                    if pred_score is not None:
                        obj_pred_scores = pred_score[ent_cate_pred_start: ent_cate_pred_end,
                                                     tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]]
                    else:
                        obj_pred_scores = torch.ones((1,))

                    obj_tok_hs = None
                    obj_int_tok_hs = None
                    if tok_hs_dump is not None:
                        obj_tok_hs = tok_hs_dump[ent_cate_pred_start: ent_cate_pred_end]
                        obj_int_tok_hs = tok_hs_dump[ent_cate_pred_end: ent_cate_pred_end+1]
                    else:
                        obj_tok_hs = torch.ones((768,)).to(self.device)
                        obj_int_tok_hs = torch.ones((768,)).to(self.device)

                    if check_label_name(obj_label_name):
                        # print("corrput obj_label_name", obj_label_name)
                        bad_inst += 1
                        curr_t_idx = curr_t_idx + 1
                        last_start = curr_t_idx
                        continue

                    ent_box_start = box_coord_b + 1
                    ent_box_end = box_coord_e
                    obj_boxs = tok_seq_nppmpt[ent_box_start: ent_box_end].detach(
                    ).cpu()
                    obj_boxs = (obj_boxs - self.coord_token_start) / \
                        self.num_coord_bins

                    # print(sub_label_name, sub_label_token, sub_pred_scores)
                    # print(predicate_label_name, predicate_label_token, predicate_pred_scores)
                    # print(obj_label_name, obj_label_token, obj_pred_scores)

                    # if sub_label_name[0] == 'and' or obj_label_name[0] == 'and':

                    new_inst = {"sub": {"label": " ".join(sub_label_name),
                                        "label_tkn": sub_label_token,
                                        "boxes": sub_boxs,
                                        "token_start": sub_ent_start,
                                        "token_end": sub_ent_end,
                                        "pred_scores": sub_pred_scores.detach().cpu(),
                                        'dec_hs': sub_tok_hs,
                                        'int_tok_hs': sub_int_tok_hs,
                                        },
                                'obj': {"label": " ".join(obj_label_name),
                                        "label_tkn": obj_label_token,
                                        "boxes": obj_boxs,
                                        "token_start": obj_ent_start,
                                        "token_end": obj_ent_end,
                                        "pred_scores": obj_pred_scores.detach().cpu(),
                                        'dec_hs': obj_tok_hs,
                                        'int_tok_hs': obj_int_tok_hs,
                                        },
                                "predicate": {"label": predicate_label_name,
                                              "label_tkn": predicate_label_token,
                                              "pred_scores": predicate_pred_scores,
                                              "token_start": pred_label_start,
                                              "token_end": pred_label_end,
                                              'dec_hs': predicate_tok_hs,
                                              }
                                }

                    object_list.append(new_inst)

                    # update index
                    last_start = last_obj_e[0] + 1
                    curr_t_idx = curr_t_idx + 1
                else:
                    curr_t_idx += 1

                if bad_inst > 50:
                    break
            # print(curr_t_idx)

            all_inst_num.append(all_inst)
            vaild_inst_nun.append(all_inst - bad_inst)

            batch_object_list.append(object_list)

        logger.info(
            f"instance det stat {np.mean(vaild_inst_nun):.1f}/{np.mean(all_inst_num):.1f}\n seq_len {np.mean(seq_len):.1f}")
        return batch_object_list

    @torch.no_grad()
    def _postprocess(self, batch_pred_instances, batch_targets_instance):
        """_summary_

        Args:
            batch_instances (list list dict): 
                batch_size, instance_num, dict(['sub', 'obj', 'predicate'])
                dict_keys(['label', 'label_tkn', 'boxes', 'token_start', 'pred_scores', 'dec_hs', 'scores_mapped', 'labels_mapped'])
                dict_keys(['label', 'label_tkn', 'pred_scores', 'dec_hs', 'scores_mapped', 'labels_mapped'])

            batch_targets_instance list, dict:
                dict_keys(['boxes', 'det_labels', 'rel_tripets', 'image_id', 'orig_size', 'size', 'det_labels_texts', 'rel_labels_texts'])

        """
        # map dict instance prediction into BoxList
        predictions = []
        ground_truths = []
        image_info = []

        all_pair_num = self.top_k_label_num
        all_predicate_num = self.top_k_predicate_label_num

        for bi in range(len(batch_pred_instances)):
            # [h, w] LAVIS/lavis/datasets/datasets/oiv6_rel_detection.py:216
            img_orig_size = batch_targets_instance[bi]['orig_size'].cpu(
            ).tolist()

            image_info.append({
                "img_orig_size": img_orig_size,
                "image_id": batch_targets_instance[bi]['image_id']
            })

            def xywh2xyxy(boxes: torch.Tensor):
                boxes_xyxy = copy.deepcopy(boxes)
                boxes_xyxy[:, :2], boxes_xyxy[:, 2:] = boxes[:, :2] - \
                    boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2
                boxes_xyxy[:, 0::2] *= img_orig_size[1]
                boxes_xyxy[:, 1::2] *= img_orig_size[0]
                return boxes_xyxy

            gt_boxes = batch_targets_instance[bi]['boxes']
            groundtruth = BoxList(
                xywh2xyxy(gt_boxes), (img_orig_size[1], img_orig_size[0]), mode="xyxy")
            groundtruth.add_field(
                'relation_tuple', batch_targets_instance[bi]['rel_tripets'])
            groundtruth.add_field(
                'labels', batch_targets_instance[bi]['det_labels'])
            ground_truths.append(groundtruth.to(torch.device('cpu')))

            inst_pred = batch_pred_instances[bi]

            if len(inst_pred) > 0:
                # ranking all relationship
                all_pred_labels = []
                all_obj_labels = []
                all_sub_labels = []

                all_obj_scores = []
                all_sub_scores = []

                all_obj_dist = []
                all_sub_dist = []

                all_obj_box = []
                all_sub_box = []

                all_predicates_scores = []
                all_triplets_scores = []
                all_predicates_dist = []

                init_inst_idx = []
                for idx, each_inst in enumerate(inst_pred):
                    obj_scores = each_inst['obj']['scores_mapped']
                    sub_scores = each_inst['sub']['scores_mapped']
                    predicate_scores = each_inst['predicate']['scores_mapped']

                    obj_labels = each_inst['obj']['labels_mapped']
                    sub_labels = each_inst['sub']['labels_mapped']
                    predicate_labels = each_inst['predicate']['labels_mapped']

                    X, Y = torch.meshgrid(torch.arange(
                        all_pair_num), torch.arange(all_pair_num), indexing='ij')
                    X = X.reshape(-1)
                    Y = Y.reshape(-1)

                    Y = Y.unsqueeze(dim=1).repeat(
                        1, all_predicate_num).reshape(-1)
                    X = X.unsqueeze(dim=1).repeat(
                        1, all_predicate_num).reshape(-1)

                    obj_labels_expanded = obj_labels[X]
                    sub_labels_expanded = sub_labels[Y]
                    # repeating: all_pair_num * all_pair_num * all_predicate_num
                    predicate_labels_expanded = predicate_labels[:all_predicate_num].repeat(
                        1, all_pair_num*all_pair_num).view(-1)

                    obj_scores_expanded = torch.index_select(
                        obj_scores, -1, obj_labels_expanded)
                    sub_scores_expanded = torch.index_select(
                        sub_scores, -1, sub_labels_expanded)
                    predicate_scores_expanded = torch.index_select(
                        predicate_scores, -1, predicate_labels_expanded)

                    # local selections
                    triplet_scores = obj_scores_expanded * \
                        sub_scores_expanded * predicate_scores_expanded
                    # triplet_scores = predicate_scores_expanded
                    trp_scrs = torch.stack(
                        (obj_scores_expanded, sub_scores_expanded, predicate_scores_expanded))
                    trp_scr, trp_idx = triplet_scores.sort(
                        dim=0, descending=True)
                    # trp_idx = trp_idx[: int(len(trp_idx)*0.8)]

                    # unexpand field initial index
                    init_indx = torch.ones(
                        all_pair_num**2 * all_predicate_num, dtype=int) * idx
                    init_inst_idx.append(init_indx[trp_idx])

                    # extend field
                    all_triplets_scores.append(triplet_scores[trp_idx])
                    all_obj_scores.append(obj_scores_expanded[trp_idx])
                    all_sub_scores.append(sub_scores_expanded[trp_idx])
                    all_predicates_scores.append(
                        predicate_scores_expanded[trp_idx])

                    all_pred_labels.append(predicate_labels_expanded[trp_idx])
                    all_obj_labels.append(obj_labels_expanded[trp_idx])
                    all_sub_labels.append(sub_labels_expanded[trp_idx])

                    # un-extend field
                    all_obj_dist.append(obj_scores)
                    all_sub_dist.append(sub_scores)
                    all_predicates_dist.append(predicate_scores)

                    all_obj_box.append(each_inst['obj']['boxes'])
                    all_sub_box.append(each_inst['sub']['boxes'])

                all_triplets_scores = torch.cat(all_triplets_scores)
                triplets_scores, triplets_indx = all_triplets_scores.sort(
                    dim=0, descending=True)
                triplets_scores = triplets_scores[:350]
                triplets_indx = triplets_indx[:350]

                all_predicates_scores_sorted = torch.cat(
                    all_predicates_scores)[triplets_indx]
                all_sub_scores_sorted = torch.cat(
                    all_sub_scores)[triplets_indx]
                all_obj_scores_sorted = torch.cat(
                    all_obj_scores)[triplets_indx]

                all_pred_labels_sorted = torch.cat(
                    all_pred_labels)[triplets_indx]
                all_obj_labels_sorted = torch.cat(
                    all_obj_labels)[triplets_indx]
                all_sub_labels_sorted = torch.cat(
                    all_sub_labels)[triplets_indx]

                # unexpand fields
                init_inst_idx_sorted = torch.cat(init_inst_idx)[triplets_indx]
                all_obj_dist_sorted = torch.stack(
                    all_obj_dist)[init_inst_idx_sorted]
                all_sub_dist_sorted = torch.stack(
                    all_sub_dist)[init_inst_idx_sorted]
                all_predicates_dist_sorted = torch.stack(all_predicates_dist)[
                    init_inst_idx_sorted]

                all_obj_box_sorted = torch.stack(
                    all_obj_box)[init_inst_idx_sorted]
                all_sub_box_sorted = torch.stack(
                    all_sub_box)[init_inst_idx_sorted]

                # fill data format
                all_box = torch.cat(
                    (all_sub_box_sorted, all_obj_box_sorted), dim=0)

                all_box = xywh2xyxy(all_box)
                invalid_idx = all_box[:, 2:] < all_box[:, :2]
                all_box[:, :2][invalid_idx] = all_box[:,
                                                      2:][invalid_idx] - 0.001
                prediction = BoxList(all_box,
                                     (img_orig_size[1], img_orig_size[0]), mode="xyxy")

                prediction.add_field('pred_labels', torch.cat(
                    (all_sub_labels_sorted, all_obj_labels_sorted), dim=0))
                prediction.add_field('pred_scores', torch.cat(
                    (all_sub_scores_sorted, all_obj_scores_sorted), dim=0))
                prediction.add_field('pred_dists', torch.cat(
                    (all_sub_dist_sorted, all_obj_dist_sorted), dim=0))

                prediction.add_field('rel_pair_idxs',
                                     torch.stack((torch.arange(len(all_sub_labels_sorted)),
                                                  torch.arange(len(all_sub_labels_sorted)) + len(all_sub_labels_sorted))
                                                 ).T  # N, 2
                                     )

                prediction.add_field(
                    'pred_rel_dist', all_predicates_dist_sorted)
                prediction.add_field(
                    'pred_rel_score', all_predicates_scores_sorted)
                prediction.add_field('pred_rel_label', all_pred_labels_sorted)
                prediction.add_field('pred_rel_trp_score', triplets_scores)
                prediction = prediction.to(torch.device('cpu'))

            else:
                #  padding instance
                prediction = BoxList(torch.zeros((1, 4)),
                                     (img_orig_size[1], img_orig_size[0]), mode="xyxy")

                prediction.add_field('pred_labels', torch.zeros((1)).int())
                prediction.add_field('pred_scores', torch.zeros((1)))
                prediction.add_field('pred_dists', torch.zeros(
                    (1, len(self.cate_dict['obj']))))

                prediction.add_field('rel_pair_idxs', torch.zeros((1, 2)))

                prediction.add_field('pred_rel_dist', torch.zeros(
                    (1, len(self.cate_dict['rel']))))
                prediction.add_field('pred_rel_score', torch.zeros((1)))
                prediction.add_field('pred_rel_label', torch.zeros((1)).int())
                prediction.add_field('pred_rel_trp_score', torch.zeros((1)))
                prediction = prediction.to(torch.device('cpu'))

            predictions.append(prediction)

        return predictions, ground_truths, image_info

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg,)
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoderDetHead.from_config(cfg,)

        # image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=True)
        # text_decoder = XBertLMHeadDecoderDetHead.from_config(cfg, from_pretrained=True)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 500)
        logger.info("model CFG for load")
        logger.info(cfg)

        model = cls(image_encoder, text_decoder, prompt=prompt, add_noise=cfg.get("add_noise", False),
                    max_txt_len=max_txt_len,
                    dump_pred=cfg.get("dump_pred", False),
                    reduction=cfg.get("reduction", 'mean'),
                    num_coord_bin=cfg.get("num_coord_bin", 1000),
                    mask_label_ratio=cfg.get("mask_label_ratio", 0.0),
                    top_k_label_num=cfg.get("top_k_predicate_label_num", 3),
                    top_k_predicate_label_num=cfg.get(
                        "top_k_predicate_label_num", 3),
                    aux_close_classifier=cfg.get(
                        "aux_close_classifier", False),
                    cate_dict_url=cfg.get("cate_dict_url", ""),
                    dump_dir=cfg.get("dump_dir", None),
                    box_loss_weight=cfg.get("box_loss_weight", 1.0),)
        # from strach
        model.load_checkpoint_from_config(cfg)

        return model




class XBertLMHeadDecoderDetHead(XBertLMHeadDecoder):

    def __init__(self, config, pos_adapter=False):
        super(XBertLMHeadDecoder, self).__init__(config)
        self.init_weights()
        hidden_dim_in = 768
        hidden_dim = 256
        self.pos_adapter_on = pos_adapter
        if pos_adapter:

            # self.conv_module = LightweightConv(hidden_dim)
            self.res18 = Res18Wrapper(out_channels=hidden_dim)

            self.enc_input_proj = nn.Linear(hidden_dim_in, hidden_dim)
            self.ent_hs_input_proj = nn.Linear(hidden_dim_in, hidden_dim)
            self.position_embedding = PositionEmbeddingSine(
                num_pos_feats=hidden_dim // 2,
                temperature=10000,
                normalize=True,
                scale=None,)
            self.pos_encoder = build_encoder(num_decoder_layers=6, d_model=hidden_dim)
            self.pos_decoder = build_decoder(num_decoder_layers=6, d_model=hidden_dim)

            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    @classmethod
    def from_config(cls, cfg, from_pretrained=False):

        med_config_path = get_abs_path(cfg.get("med_config_path"))
        med_config = BertConfig.from_json_file(med_config_path)
        vocab_size_init = med_config.vocab_size

        if med_config.add_coordinate_embeddings:
            med_config.vocab_size += (med_config.coordinates_bin +
                                      SPEICAL_TOKEN_NUM)
            logger.info(
                f"med_config.vocab_size {vocab_size_init} + bin_num:{med_config.coordinates_bin} + SPE_TKN:{SPEICAL_TOKEN_NUM} -> {med_config.vocab_size}")

        logger.info("med_config")
        logger.info(str(med_config))

        if from_pretrained:
            print("load from pretrained bert-base-uncased")
            return cls.from_pretrained("bert-base-uncased", config=med_config, ignore_mismatched_sizes=False)
        else:
            return cls(config=med_config, pos_adapter=cfg.get("pos_adapter", False))


    def pos_adapter(self, entity_hs, raw_image, encoder_hidden_states, encoder_attention_mask, box_targets=None):
        """
        encoder_hidden_states: B T D
        encoder_attention_mask:B T D

        """
        # use selected token as query for detect things from the encoder features
        max_len = max([len(each) for each in entity_hs]) + 1
        h_dim = entity_hs[0].shape[-1]
        device = encoder_hidden_states[0].device

        ent_padding_mask = torch.ones(
            len(entity_hs), max_len).bool().to(device)
        ent_hs_padded = torch.zeros(len(entity_hs), max_len, h_dim).to(device)

        #  a ``True`` value indicates that the corresponding ``key`` value will be ignored for
        for bid, ent_hs in enumerate(entity_hs):
            ent_padding_mask[bid, len(ent_hs):] = False
            ent_hs_padded[bid, :len(ent_hs), :] = ent_hs

        # todo position coding
        vis_enc_hs = encoder_hidden_states

        vis_enc_hs = self.enc_input_proj(vis_enc_hs)
        ent_hs_padded = self.ent_hs_input_proj(ent_hs_padded)

        feat_w = int(vis_enc_hs[:, 1:].shape[1] ** 0.5)
        vis_enc_map = vis_enc_hs[:, 1:].transpose(1, 0)
        vis_enc_mask = encoder_attention_mask[:, 1:].reshape(
            vis_enc_hs.shape[0], feat_w, feat_w,)
        vis_enc_mask = (1 - vis_enc_mask).bool()
        vis_enc_pos_emb = self.position_embedding(
            vis_enc_map, vis_enc_mask).flatten(2).permute(2, 0, 1)
        vis_enc_mask = vis_enc_mask.flatten(1)

        # conv_out = self.conv_module(raw_image)
        conv_out_res18 = self.res18(raw_image)
        conv_out = F.interpolate(conv_out_res18, scale_factor=2, mode="nearest")

        vis_enc_map = vis_enc_map + conv_out.flatten(-2).permute(2, 0, 1)
        encoder_hs = self.pos_encoder(  # T, B, D; B, T
            src=vis_enc_map, src_key_padding_mask=vis_enc_mask, pos=vis_enc_pos_emb
        )  # -> T, B, D
        # todo query decodeing
        ent_hs_padded = ent_hs_padded.transpose(1, 0)
        decode_ent_hs = self.pos_decoder(
            tgt=ent_hs_padded.to(device),
            memory=encoder_hs.to(device),
            tgt_key_padding_mask=ent_padding_mask.to(device),
            memory_key_padding_mask=vis_enc_mask.to(device),
            pos=vis_enc_pos_emb.to(device)
        )
        # todo pos prediction
        ent_box_pred = self.bbox_embed(
            decode_ent_hs).sigmoid().transpose(1, 0)  # B, T, 4
        # todo loss calculation
        extracted_box = []
        extracted_box_seg = [len(each) for each in entity_hs]
        for bid, each in enumerate(entity_hs):
            extracted_box.append(ent_box_pred[bid][:len(each)])

        # todo output packing
        outputs = {'extracted_box': extracted_box,
                   'extracted_box_seg': extracted_box_seg}
        if box_targets is not None:
            gt_box_cat_all = []
            for bid, box_gts in enumerate(box_targets):
                coords = []
                for bgt in box_gts:
                    box_coord = bgt['xywh']
                    coords.append(box_coord)

                coords = torch.stack(coords)
                gt_box_cat_all.append(coords)
            gt_box_cat_all = torch.cat(gt_box_cat_all)
            pred_box_cat = torch.cat(extracted_box)

            bbox_l1 = F.l1_loss(
                pred_box_cat, gt_box_cat_all, reduction='none').sum(-1).mean()
            
            bbox_giou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(pred_box_cat),
                box_cxcywh_to_xyxy(gt_box_cat_all)
            ))
            
            bbox_giou_loss = (1 - bbox_giou).mean()

            outputs['pos_adp_loss'] = {
                'pos_adp_total_loss': bbox_l1 * 2 + bbox_giou_loss * 5,
                'pos_adp_bbox_giou': bbox_giou.mean(),
                'pos_adp_bbox_l1': bbox_l1
            }
        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        box_targets=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        reduction="mean",
        mode="multimodal",
        soft_labels=None,
        alpha=0,
        model_config=None,
        box_loss_weights_all=1.0,
        raw_image=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        outputs = self.bert(  # BertModel
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
        )

        if return_dict:
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        lm_loss = None
        loss_dict = {}

        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:,
                                                          :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            token_type = get_token_type(labels)
            num_pos = (token_type == WORD).sum(-1)

            # build box loss weights
            box_l1 = []
            box_giou = []
            box_loss_weights = torch.ones_like(labels).float()
            pos_ada_output = None
            if box_targets is not None:
                box_loss_weights = torch.ones_like(labels) * 1.0
                coord_bin_range = model_config['coord_token_start'] + \
                    model_config['num_coord_bins']
                vocab_size = model_config['coord_token_start'] + 1000 + 1

                bbox_prob_mask = torch.zeros(vocab_size)
                bbox_prob_mask[model_config['coord_token_start'] +
                               1: coord_bin_range] = 1
                bbox_prob_mask = (1.0 - bbox_prob_mask) * -10000.0
                bbox_prob_mask = bbox_prob_mask.view(
                    1, 1, bbox_prob_mask.shape[-1])
                mkd_prediction_scores = prediction_scores + \
                    bbox_prob_mask.to(prediction_scores.device)

                # print(mkd_prediction_scores.shape, len(box_targets))
                pred_ent_hs_all = []
                for bid, box_gts in enumerate(box_targets):
                    pred_box_seq = (torch.argmax(mkd_prediction_scores[bid].contiguous(), dim=-1)
                                    - model_config['coord_token_start']) / model_config['num_coord_bins']  # seq_len
                    pred_boxs = []
                    gt_boxs = []
                    pred_ent_hs = []
                    for bgt in box_gts:
                        box_coord = bgt['xywh']
                        gt_boxs.append(box_coord)
                        tkn_pos = bgt['tkn_pos']
                        pred_box = pred_box_seq[tkn_pos[0]-1: tkn_pos[1]-1]
                        # xywh
                        pred_boxs.append(pred_box)
                        pred_ent_hs.append(
                            sequence_output[bid, tkn_pos[0]-2] + sequence_output[bid, tkn_pos[0]-3])
                    pred_boxs = torch.stack(pred_boxs)
                    gt_boxs = torch.stack(gt_boxs)
                    pred_ent_hs = torch.stack(pred_ent_hs)

                    pred_ent_hs_all.append(pred_ent_hs)

                    weight_bbox = F.l1_loss(
                        pred_boxs, gt_boxs, reduction='none')
                    
                    box_giou_value = torch.diag(generalized_box_iou(
                        box_cxcywh_to_xyxy(pred_boxs), box_cxcywh_to_xyxy(gt_boxs)
                    ))
                    weight_giou = (1 - box_giou_value)

                    box_l1.append(weight_bbox.sum(-1).mean().item())
                    box_giou.append(box_giou_value.mean().item())

                    loss_weight = (weight_bbox * 2) * (weight_giou * 1).unsqueeze(1)
                    for box_id, bgt in enumerate(box_gts):
                        tkn_pos = bgt['tkn_pos']
                        box_loss_weights[bid, tkn_pos[0] -
                                         1: tkn_pos[1]-1] *= loss_weight[box_id]

                if self.pos_adapter_on:

                    pos_ada_output = self.pos_adapter(
                        pred_ent_hs_all, raw_image, encoder_hidden_states, encoder_attention_mask, box_targets)

            box_loss_weights[token_type == COORD] *= box_loss_weights_all
            label_smoothing = 0.01
            if reduction == "mean":
                loss_fct = torch.nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing, reduction="none",)
                init_lm_loss = loss_fct(
                    shifted_prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )
                weights = torch.ones(init_lm_loss.shape).to(
                    init_lm_loss.device)
                weights[labels.view(-1) == -100] = 0.00

                # valid_inst_num = len(torch.nonzero(labels.view(-1) != -100))
                # valid_inst_num = len(torch.nonzero(
                #     weights.view(-1) > 0.01))

                lm_loss = init_lm_loss * weights

                weights_merge_box = weights * box_loss_weights.view(-1)
                box_w_lm_loss = init_lm_loss * weights_merge_box
                valid_inst_num = len(torch.nonzero(
                    weights_merge_box.view(-1) > 0.01))

                lm_loss_wo_box = lm_loss.sum() / valid_inst_num
                lm_loss = box_w_lm_loss.sum() / valid_inst_num

                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none", label_smoothing=label_smoothing, ignore_index=-100)
                loss_none = loss_fct(shifted_prediction_scores.permute(
                    0, 2, 1), labels)  # batch_size, seq
                loss_none_box_weighted = loss_none * box_loss_weights
                word_loss = (
                    loss_none_box_weighted[token_type == WORD].sum() / len(torch.nonzero(token_type == WORD))).mean()
                coord_loss = (
                    loss_none_box_weighted[token_type == COORD].sum() / len(torch.nonzero(token_type == COORD))).mean()
                notif_loss = (
                    loss_none_box_weighted[token_type == NOTIF].sum() / len(torch.nonzero(token_type == NOTIF))).mean()

                word_acc = sum(shifted_prediction_scores[token_type == WORD].max(-1)[
                               1] == labels[token_type == WORD]) / labels[token_type == WORD].view(-1).shape[0]
                notif_acc = sum(shifted_prediction_scores[token_type == NOTIF].max(-1)[
                                1] == labels[token_type == NOTIF]) / labels[token_type == NOTIF].view(-1).shape[0]
                coord_error = sum(torch.abs(shifted_prediction_scores[token_type == COORD].max(-1)[
                                  1] - labels[token_type == COORD])) / (labels[token_type == COORD].view(-1).shape[0] * 4)
            if reduction == "none":
                num_pos = (token_type == WORD).sum(-1)
                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none", label_smoothing=label_smoothing, ignore_index=-100)

                lm_loss_wo_box = loss_fct(shifted_prediction_scores.permute(
                    0, 2, 1), labels)  # batch_size, seq
                lm_loss = lm_loss_wo_box * box_loss_weights.view(-1)

                weights = torch.ones(labels.shape).to(loss_none.device)
                weights[labels == -100] = 0.00  # padding
                weighted_lm_loss = lm_loss * weights
                lm_loss = (weighted_lm_loss.sum(-1) / num_pos).mean()

                word_loss = weighted_lm_loss[token_type == WORD].sum(
                ) / (num_pos * labels.shape[0])
                coord_loss = weighted_lm_loss[token_type == COORD].sum(
                ) / (num_pos * labels.shape[0])
                notif_loss = weighted_lm_loss[token_type == NOTIF].sum(
                ) / (num_pos * labels.shape[0])

                word_acc = sum(shifted_prediction_scores[token_type == WORD].max(-1)[
                               1] == labels[token_type == WORD]) / weighted_lm_loss[token_type == WORD].view(-1).shape[0]
                notif_acc = sum(shifted_prediction_scores[token_type == NOTIF].max(-1)[
                                1] == labels[token_type == NOTIF]) / weighted_lm_loss[token_type == NOTIF].view(-1).shape[0]
                coord_error = sum(torch.abs(shifted_prediction_scores[token_type == COORD].max(-1)[
                                  1] - labels[token_type == COORD])) / (weighted_lm_loss[token_type == COORD].view(-1).shape[0] * 4 * 1000)

            word_loss = word_loss.mean()
            coord_loss = coord_loss.mean()
            notif_loss = notif_loss.mean()

            box_l1 = torch.Tensor(box_l1)
            box_giou = torch.Tensor(box_giou)
            loss_dict = {
                'loss': lm_loss,
                'init_lm_loss': lm_loss_wo_box,
                'word_loss': word_loss,
                "coord_loss": coord_loss,
                "notif_loss": notif_loss,
                "notif_acc": notif_acc,
                "word_acc": word_acc,
                'coord_error': coord_error,
                'box_l1_error': torch.mean(box_l1),
                'box_giou': torch.mean(box_giou)
            }
            if pos_ada_output is not None:
                loss_dict['loss'] += pos_ada_output['pos_adp_loss']['pos_adp_total_loss']
                loss_dict.update(pos_ada_output['pos_adp_loss'])

        if soft_labels is not None:
            loss_distill = -torch.sum(
                F.log_softmax(shifted_prediction_scores, dim=-1) * soft_labels, dim=-1
            )
            loss_distill = (loss_distill * (labels != -100)).sum(1)
            lm_loss = (1 - alpha) * lm_loss + alpha * loss_distill

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss_dict,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_states=sequence_output,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def generate_from_encoder(
        self,
        tokenized_prompt,
        visual_embeds,
        sep_token_id,
        pad_token_id,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_return_sequences=1,
        **kwargs
    ):

        if not use_nucleus_sampling:
            num_beams = num_beams
            visual_embeds = visual_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = None

        image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        model_kwargs = {
            "encoder_hidden_states": visual_embeds,
            "encoder_attention_mask": image_atts,
        }
        model_kwargs.update(kwargs)
        # print(max_length, min_length)
        outputs = self.generate(
            input_ids=tokenized_prompt.input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            eos_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=1.5,
            **model_kwargs
        )

        return outputs


def get_token_type(labels):
    token_types = torch.zeros_like(labels)
    token_types[torch.logical_and(labels >= 103, labels < 30524)] = WORD
    token_types[torch.logical_and(labels >= 30524, labels < 30534)] = NOTIF
    token_types[labels >= 30534] = COORD
    return token_types


def breakpoint():
    import ipdb
    ipdb.set_trace(context=15)

# breakpoint()
