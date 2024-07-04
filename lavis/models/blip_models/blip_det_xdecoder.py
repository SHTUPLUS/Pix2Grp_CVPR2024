"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from pprint import pprint
import copy
import json
import time
import pickle
import os
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.ops import boxes as box_ops
import tqdm
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.bert.configuration_bert import BertConfig
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


from lavis.common.dist_utils import get_world_size
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.datasets.datasets.utils.box_ops import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh, generalized_box_iou
from lavis.models.blip_models.blip_det import BlipDetection, check_label_name, find_token
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import SPEICAL_TOKEN_NUM, XBertLMHeadDecoder
from lavis.models.resnet import BasicBlock, BasicStem, LightweightConv, Res18Wrapper
from lavis.models.vit import VisionTransformerEncoder
from lavis.models.detr_transformer import build_decoder, build_encoder, PositionEmbeddingSine, MLP
from lavis.models.weight_init import show_params_status
from lavis.tasks.evaluation.boxlist import BoxList
from lavis.models.blip_models.blip_rel_det import BlipRelDetection

SPEC = -2
WORD = -3
COORD = -4
NOTIF = -5

logger = logging.getLogger(__name__)


@registry.register_model("blip_detection_xdecoder")
class BlipDetectionXDecoder(BlipRelDetection):
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
                 lora_enable=False, seg_len=24, post_proc_cfg=None,
                 dump_dir='/mnt/petrelfs/lirongjie/project/LAVIS/lavis/output/BLIP/vis_dump'):
        super(BlipDetectionXDecoder, self).__init__(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len,
                                                       num_coord_bin=num_coord_bin, reduction=reduction,
                                                       max_objects=max_objects, max_pos_objects=max_pos_objects,
                                                       add_noise=add_noise, dump_pred=dump_pred,
                                                       top_k_label_num=top_k_label_num, top_k_predicate_label_num=top_k_predicate_label_num,
                                                       mask_label_ratio=mask_label_ratio, aux_close_classifier=aux_close_classifier,
                                                       cate_dict_url=cate_dict_url, box_loss_weight=box_loss_weight, dump_dir=dump_dir)

        self.logit_wrappers = LogitsWarper(top_k=60, min_tokens_to_keep=1, top_p=1.0)
        self.seg_len = seg_len
        if post_proc_cfg is not None:
            self.post_proc_cfg = post_proc_cfg 
        else:
            self.post_proc_cfg = {}

        self.frequency_info = {}
        if lora_enable:
            lora_r = 32
            lora_alpha = 16
            # scalering: lora_alpha / r
            lora_dropout = 0.05
            lora_target_modules = [
                "value",
                "query",
                "key",
                # "dense",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.text_decoder = get_peft_model(self.text_decoder, config)
            for name, param in self.text_decoder.named_parameters():
                if 'bbox_embed' in name:
                    param.requires_grad = True
                elif 'pos_decoder' in name:
                    param.requires_grad = True   
                elif 'pos_encoder' in name:
                    param.requires_grad = True   
                elif 'ent_hs_input_proj' in name:
                    param.requires_grad = True   
                elif 'enc_input_proj' in name:
                    param.requires_grad = True   
                elif 'cls.predictions' in name:
                    param.requires_grad = True 

        self.start_emb_weight = None

        logger.info(show_params_status(self))


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

            all_obj_pred_logits = self.text_decoder.cls.predictions(
                all_obj_pred_hs)
            all_sub_pred_logits = self.text_decoder.cls.predictions(
                all_sub_pred_hs)
            all_pred_pred_logits = self.text_decoder.cls.predictions(
                all_pred_pred_hs)

            # cls_weight = self.text_decoder.cls.predictions.decoder.weight
            # cls_bias = self.text_decoder.cls.predictions.decoder.bias
            # sub_cos_dist = F.normalize(
            #     all_sub_pred_hs) @ F.normalize(cls_weight).T

            # mimic the score distribution of LM token predictor
            all_obj_pred_logits = self.logit_wrappers(
                all_obj_pred_logits, self.ent_label_token)
            all_sub_pred_logits = self.logit_wrappers(
                all_sub_pred_logits, self.ent_label_token)
            all_pred_pred_logits = self.logit_wrappers(
                all_pred_pred_logits, self.rel_label_token)

            all_obj_pred_scores = all_obj_pred_logits.softmax(dim=-1)
            all_sub_pred_scores = all_sub_pred_logits.softmax(dim=-1)
            all_pred_pred_scores = all_pred_pred_logits.softmax(dim=-1)

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
                #  vocab 2 categories, vocabulary 多个词对应一个类别，将其average 压缩
                # num_seq, num_vocab -> num_seq, num_class
                'obj': range_sum(obj_pred_scores, ent_cls_token_range),
                'sub': range_sum(sub_pred_scores, ent_cls_token_range),
                'predicate': range_sum(pred_pred_scores, pred_cls_token_range)
            }

            token_range_all = {
                #  sequence 2 instance, sequence中多个token 对应一个instance的entity，将其average 压缩
                'obj': obj_pred_token_range,
                'sub': sub_pred_token_range,
                'predicate': predicate_token_range
            }
            # token2inst: num_seq, num_class -> num_inst, num_class
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
                    # if fld_name != 'predicate':
                    #     print(pred_instances[bi][inst_id][fld_name]['label'],
                    #         self.cate_dict['obj'][pred_instances[bi][inst_id][fld_name]['labels_mapped'][0]],
                    #         self.cate_dict['obj'][pred_instances[bi][inst_id][fld_name]['labels_mapped'][1]],
                    #         self.cate_dict['obj'][pred_instances[bi][inst_id][fld_name]['labels_mapped'][2]])
                    # else:
                    #     print(pred_instances[bi][inst_id][fld_name]['label'],
                    #         self.cate_dict['rel'][pred_instances[bi][inst_id][fld_name]['labels_mapped'][0]],
                    #         self.cate_dict['rel'][pred_instances[bi][inst_id][fld_name]['labels_mapped'][1]],
                    #         self.cate_dict['rel'][pred_instances[bi][inst_id][fld_name]['labels_mapped'][2]])

        return pred_instances

    def forward_decoder(self, samples, image_embeds, return_close_vocab_classes=False, return_box_gts_batch=False):
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
        if return_box_gts_batch:
            return decoder_output, decoder_targets, box_gts_batch
        return decoder_output, decoder_targets

    def target2token_seqs(self, targets, text_mask_ratio=0.5):

        add_noise = self.add_noise
        tar_seqs = []
        # construct the templets
        batch_num_noise = []
        for b_i, target in enumerate(targets):
            # dict_keys(['boxes', 'det_labels', 'rel_tripets', 'image_id', 'orig_size', 'size', 'det_labels_texts', 'rel_labels_texts'])

            all_idx = torch.randperm(len(target['boxes'])).tolist()

            if len(all_idx) > self.max_objects:
                all_idx = all_idx[:self.max_objects]

            target['boxes'] = target['boxes'][all_idx]
            labels_texts = [target['labels_texts'][i] for i in all_idx]
            target['det_labels'] = target['det_labels'][all_idx]
            det_labels = target['det_labels']

            rel_tri_text = []
            for det_label in target['det_labels']:
                rel_tri_text.append(
                    f"{self.cate_dict['obj'][det_label]} [OBJ]")

            seperate_word = " and "
            # print(rel_tri_text)
            # seperate_word = " , "
            tar_seq = self.prompt + seperate_word.join(rel_tri_text)

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
            det_labels = target['det_labels']
            boxes_xywh = copy.deepcopy(boxes)
            rel_idx = 0

            label_token_fuse = label_token['input_ids'][b_i, :]
            close_vocab_cate_id = []
            box_gt = []
            for det_idx, each_det_label in enumerate(det_labels):
                close_vocab_cate_id.append((each_det_label,))

                coord_id = 0
                while coord_id < 4 and rel_idx < len(label_token_fuse):
                    if label_token_fuse[rel_idx] == self.tokenizer.obj_s_token_id:
                        box_gt.append({
                            'xywh': boxes_xywh[det_idx],
                            'tkn_pos': rel_idx
                        })
                        coord_id += 4
                    rel_idx += 1

            box_gts_batch.append(box_gt)
            raw_token.append(label_token_fuse)
            close_vocab_classes.append(close_vocab_cate_id)

        raw_token = torch.stack(raw_token)
        label_token.input_ids = raw_token

        # prepare targets for forwarding decoder
        decoder_targets = label_token.input_ids.masked_fill(
            label_token.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

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

        self.frequency_info = samples.get('freq_info')

        image_embeds = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(
            samples, image_embeds)
        # if self.start_emb_weight is None:
        #     self.start_emb_weight = self.text_decoder.bert.embeddings.word_embeddings.weight
        # else:
        #     logger.info(f"embedding weight change {torch.abs(self.start_emb_weight - self.text_decoder.bert.embeddings.word_embeddings.weight).sum()}")

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
        self.frequency_info = samples.get('freq_info')

        encoder_out = self.forward_encoder(samples)
        # image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)
        image_embeds = encoder_out

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
        decoder_out_collect = None
        min_seq_len = 24
        print(max_length // min_seq_len)
        for i in range(max_length // min_seq_len):
            decoder_out = self.text_decoder.generate_from_encoder(
                tokenized_prompt=prompt,
                visual_embeds=image_embeds,
                sep_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_nucleus_sampling=use_nucleus_sampling,
                num_beams=num_beams,
                max_length=self.seg_len,
                min_length=min_seq_len,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                output_attentions=True,  # 16, 12, 192, 12, 4, 4
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                num_return_sequences=1,
            )
            # append obj token as last for eos
            # vocab_size = decoder_out.scores[0].shape[-1]
            # bz = decoder_out.sequences.shape[1]

            # obj_hs = self.text_decoder.cls.predictions.decoder.weight[30525]
            # obj_hs_token = torch.Tensor([30525]).long()
            # scores = torch.zeros((bz, vocab_size))
            # scores[:,30525] = 1.0

            # decoder_out.decoder_hidden_states = decoder_out.decoder_hidden_states + ((obj_hs.view(1,1,obj_hs.shape[-1]), ), )
            # decoder_out.sequences = torch.cat((
            #     decoder_out.sequences, obj_hs_token.view(1, 1).repeat(bz,1).to(self.device)), dim=-1)
            # decoder_out.scores = decoder_out.scores + (scores, )

            if decoder_out_collect is None:
                decoder_out_collect = decoder_out
            else:
                decoder_out_collect.hidden_states = decoder_out_collect.hidden_states + \
                    decoder_out.hidden_states
                sequences_wo_propmt = decoder_out.sequences[:,
                                                            self.prompt_length:]
                decoder_out_collect.sequences = torch.cat(
                    (decoder_out_collect.sequences, sequences_wo_propmt), dim=-1)
                decoder_out_collect.scores = decoder_out_collect.scores + decoder_out.scores

        decoder_out = decoder_out_collect

        # odict_keys(['sequences', 'decoder_attentions', 'cross_attentions', decoder_hidden_states])
        # decoder_out.cross_attentions:
        #       seq_len, decoder_layer, batch_size*num_captions, head_num, previous_seq_size(prompt_size or 1), image_hs_size
        # decoder_out.sequences:
        #       batch_size, generate_len
        # decoder_out.decoder_hidden_states:
        #       seq_len, decoder_layer, batch_size*num_beams, generate_len (第一个token因为是prompt大于1 其他都等于1), hid_dim
        #  decoder_out.scores:
        #       generate_len, batch_size*num_beams, vocab_size
        # raw_caption = [self.tokenizer.decode(decoder_out_collect.sequences[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
        #                for i in range(len(decoder_out.sequences))]

        raw_caption = [self.tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=False)
                        for seq in decoder_out.sequences]
        batch_object_list = self.seq2instance(decoder_out, raw_caption, verbose=False)

        batch_object_list_cate_trans = self.vocab2category(batch_object_list)

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
                        ent_hs.append(
                            each_box_pred[each_role]['int_tok_hs'][-1] + each_box_pred[each_role]['dec_hs'][-1])
                pred_ent_hs_all.append(torch.stack(ent_hs))
            pos_ada_output = self.text_decoder.pos_adapter(
                pred_ent_hs_all, samples['image'], image_embeds, encoder_attention_mask, None)

            role_list = ['sub', 'obj']
            for bid, adapter_box_pred in enumerate(pos_ada_output['extracted_box']):
                for inst_id, each_box_pred in enumerate(adapter_box_pred):
                    batch_object_list_cate_trans[bid][int(
                        inst_id//2)][role_list[inst_id % 2]]['boxes'] = each_box_pred

        predictions, ground_truths, image_info = self._postprocess(
            batch_object_list_cate_trans, samples['targets'])

        if self.dump_pred:
            forward_decoder_output, decoder_targets, box_targets = self.forward_decoder(
                samples, image_embeds, return_box_gts_batch=True)
            raw_caption_target = [self.tokenizer.decode(decoder_targets[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                  for i in range(len(decoder_targets))]
            f_dec_tokens = torch.argmax(
                forward_decoder_output.logits.contiguous(), dim=-1)
            raw_caption_fdec = [self.tokenizer.decode(f_dec_tokens[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                for i in range(len(f_dec_tokens))]

            pos_ada_output = None
            if self.text_decoder.pos_adapter_on:
                pred_ent_hs_all = []
                sequence_output = forward_decoder_output.hidden_states

                for bid, box_gts in enumerate(box_targets):
                    pred_ent_hs = []
                    for bgt in box_gts:
                        tkn_pos = bgt['tkn_pos']
                        # xywh
                        pred_ent_hs.append(
                            sequence_output[bid, tkn_pos-1] + sequence_output[bid, tkn_pos-2])  # obj token + categories token
                    pred_ent_hs = torch.stack(pred_ent_hs)
                    pred_ent_hs_all.append(pred_ent_hs)

                pos_ada_output = self.text_decoder.pos_adapter(
                    pred_ent_hs_all, samples['image'], image_embeds, encoder_attention_mask, box_targets)

            for bi, b_pred_inst in enumerate(batch_object_list_cate_trans):
                for each_inst in b_pred_inst:
                    each_inst['obj'].pop('dec_hs')
                    each_inst['sub'].pop('dec_hs')
                    each_inst['predicate'].pop('dec_hs')
                    each_inst['obj'].pop('int_tok_hs')
                    each_inst['sub'].pop('int_tok_hs')

                    each_inst['predicate'].pop('pred_dist')
                    each_inst['obj'].pop('pred_dist')
                    each_inst['sub'].pop('pred_dist')

            image_ids = samples['instance_id'][0]
            forward_buffer = {
                # 'cross_attentions': decoder_out['cross_attentions'],
                # 'decoder_attentions': decoder_out['decoder_attentions'],
                'scores': decoder_out['scores'],  # num_toke, bz, vocab_size
                # 'normed_image': samples['image'],
                'raw_token': decoder_out.sequences.detach().cpu(),
                # "forward_decoder_output": forward_decoder_output,
                "f_dec_tokens": f_dec_tokens,
                "f_dec_pos_ada_output": pos_ada_output,

                'raw_caption_fdec': raw_caption_fdec,
                'raw_caption': raw_caption,
                "raw_caption_target": raw_caption_target,
                "decoder_targets": decoder_targets,
                # "batch_target_object_list": batch_target_object_list,

                "predictions": predictions,
                "ground_truths": ground_truths,

                'image_path': samples['image_pth'],
                "batch_object_list": batch_object_list_cate_trans,
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

    def seq2instance(self, decoder_out, raw_caption, prompt_length=None, verbose=False):
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
            if decoder_out.get('hidden_states') is not None:
                tok_hs_dump = []
                for l_i, l in enumerate(decoder_out['hidden_states']):
                    tok_hs_dump.append(l[-1][bi][-1])
                tok_hs_dump = torch.stack(tok_hs_dump)  # remove prompt

            pred_score = None
            if decoder_out.get('scores') is not None:
                scores_tensor = []
                for each in decoder_out['scores']:
                    scores_tensor.append(each[bi].softmax(dim=-1))
                pred_score = torch.stack(scores_tensor)  # num_tok, vocab_size

            object_list = []
            curr_t_idx = 0

            bad_inst = 0
            all_inst = 0

            rel_markers = torch.zeros(len(seq_no_pmpt)).bool()
            valid_rel_markers = torch.zeros(len(seq_no_pmpt)).bool()

            while curr_t_idx < len(seq_no_pmpt):
                tok_word = seq_no_pmpt[curr_t_idx]
                if tok_word.upper() == '[REL]':
                    if verbose:
                        start_p = 0 if curr_t_idx-5 < 0 else curr_t_idx-5
                        print(start_p, curr_t_idx+5)
                        print(seq_no_pmpt[start_p: curr_t_idx+6])
                        print(tok_seq_nppmpt[start_p: curr_t_idx+6])

                    rel_markers[curr_t_idx] = True

                    all_inst += 1
                    # find subject

                    start_p = 0 if curr_t_idx-15 < 0 else curr_t_idx-15
                    last_obj_b = find_token(
                        seq_no_pmpt, ['[obj]', '[pad]', '[sep]'], start_p, curr_t_idx)

                    if len(last_obj_b) < 1:
                        bad_inst += 1
                        # update index
                        curr_t_idx = curr_t_idx + 1
                        if verbose:
                            print('faild on find obj token', last_obj_b)
                        continue
                    last_obj_b = [-1,] + last_obj_b

                    ent_cate_pred_end = last_obj_b[-1]
                    ent_cate_pred_start = last_obj_b[-2] + 1

                    if seq_no_pmpt[ent_cate_pred_start] == 'and' or seq_no_pmpt[ent_cate_pred_start] == ',':
                        ent_cate_pred_start += 1
                    sub_ent_start = ent_cate_pred_start
                    sub_ent_end = ent_cate_pred_end

                    sub_label_name = seq_no_pmpt[ent_cate_pred_start: ent_cate_pred_end]
                    sub_label_token = tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]
                    if pred_score is not None:
                        sub_pred_scores = pred_score[torch.arange(
                            ent_cate_pred_start, ent_cate_pred_end), sub_label_token]
                        sub_pred_scores_dist = pred_score[ent_cate_pred_start: ent_cate_pred_end]
                    else:
                        sub_pred_scores = torch.ones((1,))
                        sub_pred_scores_dist = torch.ones((1,))

                    sub_tok_hs = None
                    sub_int_tok_hs = None
                    if tok_hs_dump is not None:
                        sub_tok_hs = tok_hs_dump[ent_cate_pred_start: ent_cate_pred_end]
                        sub_int_tok_hs = tok_hs_dump[ent_cate_pred_end: ent_cate_pred_end+1]
                    else:
                        sub_tok_hs = torch.ones((768,)).to(self.device)
                        sub_int_tok_hs = torch.ones((768,)).to(self.device)

                    if check_label_name(sub_label_name):
                        # print("corrput sub_label_name", sub_label_name)
                        bad_inst += 1
                        # update index
                        curr_t_idx = curr_t_idx + 1
                        if verbose:
                            print('invalid sub name', sub_label_name)
                        continue

                    # get predicate prediction
                    pred_label_start = ent_cate_pred_end + 1
                    pred_label_end = curr_t_idx
                    predicate_label_name = seq_no_pmpt[pred_label_start: pred_label_end]
                    predicate_label_token = tok_seq_nppmpt[pred_label_start: pred_label_end]
                    if check_label_name(predicate_label_name):
                        # print("corrput predicate_label_name",predicate_label_name)
                        bad_inst += 1
                        curr_t_idx = curr_t_idx + 1
                        if verbose:
                            print('invalid predicate name', predicate_label_name)
                        continue

                    predicate_pred_scores = None
                    if pred_score is not None:
                        predicate_pred_scores = pred_score[torch.arange(
                            pred_label_start, pred_label_end), predicate_label_token]
                        predicate_pred_scores_dist = pred_score[pred_label_start: pred_label_end]
                    else:
                        predicate_pred_scores = torch.ones((1,))
                        predicate_pred_scores_dist = torch.ones((1,))

                    predicate_tok_hs = None
                    if tok_hs_dump is not None:
                        predicate_tok_hs = tok_hs_dump[pred_label_start: pred_label_end]
                    else:
                        predicate_tok_hs = torch.ones((768,)).to(self.device)

                    # get object
                    find_end = len(seq_no_pmpt) if curr_t_idx + \
                        8 > len(seq_no_pmpt) else curr_t_idx+8
                    last_obj_b = find_token(
                        seq_no_pmpt, ['[obj]', '[pad]', '[sep]'], curr_t_idx, find_end)

                    if len(last_obj_b) < 1:
                        bad_inst += 1
                        # update index
                        curr_t_idx = curr_t_idx + 1
                        if verbose:
                            print('can not find obj ent token', last_obj_b)
                        
                        continue
                    last_obj_b.append(len(seq_no_pmpt)-1)

                    ent_cate_pred_start = pred_label_end + 1
                    ent_cate_pred_end = last_obj_b[0]

                    obj_ent_start = ent_cate_pred_start
                    obj_ent_end = ent_cate_pred_end

                    obj_label_name = seq_no_pmpt[ent_cate_pred_start: ent_cate_pred_end]
                    obj_label_token = tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]

                    if pred_score is not None:
                        obj_pred_scores = pred_score[torch.arange(
                            ent_cate_pred_start, ent_cate_pred_end), obj_label_token]
                        obj_pred_dist = pred_score[ent_cate_pred_start: ent_cate_pred_end]
                    else:
                        obj_pred_scores = torch.ones((1,))
                        obj_pred_dist = torch.ones((1,))

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

                        if verbose:
                            print('invaild object name', obj_label_name)
                        continue

                    # if sub_label_name[0] == 'and' or obj_label_name[0] == 'and':
                    # print(self.tokenizer.clean_text_from_decode(sub_label_name), " ".join(sub_label_name))

                    new_inst = {"sub": {
                                        "label": self.tokenizer.clean_text_from_decode(sub_label_name),
                                        # "label": " ".join(sub_label_name),
                                        "label_tkn": sub_label_token,
                                        "token_start": sub_ent_start,
                                        "token_end": sub_ent_end,
                                        "pred_scores": sub_pred_scores.detach().cpu(),
                                        'pred_dist': sub_pred_scores_dist,
                                        'dec_hs': sub_tok_hs,
                                        'int_tok_hs': sub_int_tok_hs, },
                                'obj': {
                                        "label": self.tokenizer.clean_text_from_decode(obj_label_name),
                                        # "label": " ".join(obj_label_name),
                                        "label_tkn": obj_label_token,
                                        "token_start": obj_ent_start,
                                        "token_end": obj_ent_end,
                                        "pred_scores": obj_pred_scores.detach().cpu(),
                                        'pred_dist': obj_pred_dist,
                                        'dec_hs': obj_tok_hs,
                                        'int_tok_hs': obj_int_tok_hs, },
                                "predicate": {
                                            "label": self.tokenizer.clean_text_from_decode(predicate_label_name),
                                            #  "label": " ".join(obj_label_name),
                                              "label_tkn": predicate_label_token,
                                              "pred_scores": predicate_pred_scores,
                                              'pred_dist': predicate_pred_scores_dist,
                                              "token_start": pred_label_start,
                                              "token_end": pred_label_end,
                                              'dec_hs': predicate_tok_hs, }}
                    if verbose:
                        print(new_inst['sub']['label'], new_inst['predicate']['label'], new_inst['obj']['label'])
                        print(sub_label_name, sub_label_token, sub_pred_scores)
                        print(predicate_label_name, predicate_label_token, predicate_pred_scores)
                        print(obj_label_name, obj_label_token, obj_pred_scores)


                    object_list.append(new_inst)
                    valid_rel_markers[curr_t_idx] = True

                    # update index
                    curr_t_idx = curr_t_idx + 1
                else:
                    curr_t_idx += 1

            # unmatched_idx = torch.abs(rel_markers.int() - valid_rel_markers.int()).nonzero().view(-1)
            # print("all_inst", len(object_list), "unmatched_idx", len(unmatched_idx), unmatched_idx)

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

        show_match_detail = False

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
                    obj_scores = each_inst['obj']['scores_mapped'].cpu()
                    sub_scores = each_inst['sub']['scores_mapped'].cpu()
                    predicate_scores = each_inst['predicate']['scores_mapped'].cpu(
                    )

                    obj_pred_scores = each_inst['obj']['pred_scores'].cpu(
                    ).max()
                    sub_pred_scores = each_inst['sub']['pred_scores'].cpu(
                    ).max()
                    predicate_pred_scores = each_inst['predicate']['pred_scores'].cpu(
                    ).max()

                    obj_pred_dist = each_inst['obj']['pred_dist'].cpu()
                    sub_pred_dist = each_inst['sub']['pred_dist'].cpu()
                    predicate_pred_dist = each_inst['predicate']['pred_dist'].cpu(
                    )

                    obj_labels = each_inst['obj']['labels_mapped'].cpu()
                    sub_labels = each_inst['sub']['labels_mapped'].cpu()
                    predicate_labels = each_inst['predicate']['labels_mapped'].cpu(
                    )


                    # mimic the sampling behavior of sequence generation
                    def dump_sampling_res(cls_score, sample_scr, cate_dict, gen_words):
                        gen_match_score, init_gen_idx = torch.abs(
                            cls_score - sample_scr).min(-1)
                        gen_idx = init_gen_idx.unsqueeze(0).clone() # [idx]
                        
                        idx_by_cate_name = index(cate_dict, gen_words)

                        matched = True
                        if gen_idx.item() != idx_by_cate_name:
                            matched = False
                            if idx_by_cate_name != -1:
                                gen_idx = torch.Tensor([idx_by_cate_name]).long()
                            else:
                                # gen_idx = init_gen_idx
                                gen_idx = None
                        return gen_idx, init_gen_idx, matched

                    sub_gen_idx, init_sub_gen_idx, sub_matched = dump_sampling_res(
                        sub_scores, sub_pred_scores, self.cate_dict['obj'], each_inst['sub']['label'])
                    obj_gen_idx, init_obj_gen_idx, obj_matched = dump_sampling_res(
                        obj_scores, obj_pred_scores, self.cate_dict['obj'], each_inst['obj']['label'])
                    predicate_gen_idx, init_predicate_gen_idx, pred_matched = dump_sampling_res(
                        predicate_scores, predicate_pred_scores, self.cate_dict['rel'], each_inst['predicate']['label'])

                    init_obj_cate_id = index(
                        self.cate_dict['obj'], each_inst['obj']['label'])
                    init_sub_cate_id = index(
                        self.cate_dict['obj'], each_inst['sub']['label'])
                    init_pred_cate_id = index(
                        self.cate_dict['rel'], each_inst['predicate']['label'])

                    def topk_score_selected(select_range, pred_probs, top_p_filter=False, top_k_filter=False,
                                            freq_adjust=None, freq_adjust_gamma=0.8, topk=15, top_p=1.0,
                                            gen_idx=None, ampl_scale=5.0, temperature=1.0):
                        filter_value = 1e-4

                        if freq_adjust is not None:
                            freq_adjust = freq_adjust.to(pred_probs.device)
                            pred_probs_la = (
                                pred_probs.log() - freq_adjust_gamma * freq_adjust.log()).softmax(dim=-1)
                            pred_probs = pred_probs_la

                        if top_k_filter:
                            indices_to_remove = pred_probs < torch.topk(pred_probs, topk)[
                                0][..., -1, None]
                            pred_probs = pred_probs.masked_fill(
                                indices_to_remove, filter_value)

                        # prob renormalization 

                        # logits = pred_probs.log()
                        # logits[pred_probs < filter_value] = float('-inf')
                        # pred_probs = logits.softmax(-1)

                        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                        if top_p_filter:
                            sorted_probs, sorted_indices = torch.sort(
                                pred_probs, descending=True)
                            cumulative_probs = sorted_probs.cumsum(dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Shift the indices to the right to keep also the first token above the threshold
                            sorted_indices_to_remove[...,
                                                     1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            # scatter sorted tensors to original indexing
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                0, sorted_indices, sorted_indices_to_remove)
                            pred_probs = pred_probs.masked_fill(
                                indices_to_remove, filter_value)

                        pred_scores = pred_probs.clone()
                        ###########################################
                        # 抑制非选中index的score
                        # if gen_idx is not None:
                        #     head_weight = pred_probs.sum() * ampl_scale
                        #     samp_weight_probs = pred_probs * 0.5
                        #     samp_weight_probs[gen_idx] = head_weight

                        # pred_labels = torch.multinomial(
                        #     samp_weight_probs, num_samples=select_range)
                        
                        # # temperature = self.post_proc_cfg.get('temperature', 1.0)
                        # logits = samp_weight_probs.log() 
                        # logits[samp_weight_probs < filter_value * 2] = float('-inf')
                        # pred_scores = (logits / temperature).softmax(-1)

                        ###########################################
                        # 只增强选中index
                        if gen_idx is not None:
                            # only amplify matched idx
                            pred_scores[gen_idx] = pred_scores[gen_idx] * ampl_scale
                            
                            if select_range > 1:
                                samp_weight_probs_tmp = pred_scores.clone()
                                samp_weight_probs_tmp[gen_idx] = 0.0
                                ext_pred_labels = torch.multinomial(samp_weight_probs_tmp, num_samples=select_range-1)
                                pred_labels = torch.cat((gen_idx, ext_pred_labels), dim=0)

                            else:
                                pred_labels = gen_idx
                        else:
                            pred_labels = torch.multinomial(pred_scores, num_samples=select_range)
                        samp_weight_probs = pred_scores
                        ###########################################

                        return pred_labels, pred_scores, samp_weight_probs

                    def topk_score_selected_simple(select_range, pred_probs, gen_idx=None, ampl_scale=5.0, temperature=1.0):

                        # prob renormalization 
                        filter_value = 1e-4

                        # logits = pred_probs.log()
                        # logits[pred_probs < filter_value] = float('-inf')
                        # pred_probs = logits.softmax(-1)

                        pred_scores = pred_probs.clone()

                        # 只增强选中index
                        if gen_idx is not None:
                            # only amplify matched idx
                            pred_scores[gen_idx] = pred_scores[gen_idx] * ampl_scale
                            
                            if select_range > 1:
                                samp_weight_probs_tmp = pred_scores.clone()
                                samp_weight_probs_tmp[gen_idx] = 0.0
                                ext_pred_labels = torch.multinomial(samp_weight_probs_tmp, num_samples=select_range-1)
                                pred_labels = torch.cat((gen_idx, ext_pred_labels), dim=0)

                            else:
                                pred_labels = gen_idx
                        else:
                            pred_labels = torch.multinomial(pred_scores, num_samples=select_range)
                        samp_weight_probs = pred_scores
                        return pred_labels, pred_scores, samp_weight_probs
                    
                    def sample_labels(select_range, probs):
                        pred_labels = torch.multinomial(
                            probs, num_samples=select_range, replacement=False)
                        return pred_labels, probs
                    
                    #  find matching detection results
                    # obj_box = box_cxcywh_to_xyxy(each_inst['obj']['boxes'].cpu()[None, :]) # xyxy_abs
                    # sub_box = box_cxcywh_to_xyxy(each_inst['sub']['boxes'].cpu()[None, :])
                    # sub_match = generalized_box_iou(box_cxcywh_to_xyxy(gt_boxes).cpu(), sub_box) # num_gt, 1
                    # obj_match = generalized_box_iou(box_cxcywh_to_xyxy(gt_boxes).cpu(), obj_box)
                    # for gt_rel in batch_targets_instance[bi]['rel_tripets']:
                    #     gt_rel_list = gt_rel.tolist()
                    #     if obj_match[gt_rel_list[1]] > 0.5 and sub_match[gt_rel_list[0]] > 0.5:
                    #         print(gt_rel)
                    #         print("prediction match gt")
                    #         import ipdb; ipdb.set_trace(context=20)


                    # import ipdb; ipdb.set_trace(context=20)
                    # show_match_detail = True
                    # obj_labels, obj_scores, obj_sample_prob = topk_score_selected(all_pair_num, obj_scores, gen_idx=obj_gen_idx,
                    #                                                               ampl_scale=self.post_proc_cfg.get('ent_ampl_scale', 4.0))
                    # sub_labels, sub_scores, sub_sample_prob = topk_score_selected(all_pair_num, sub_scores, gen_idx=sub_gen_idx,
                    #                                                               ampl_scale=self.post_proc_cfg.get('ent_ampl_scale', 4.0)) 
                    # predicate_labels, predicate_scores, predi_sample_prob = topk_score_selected(all_predicate_num, predicate_scores,gen_idx=predicate_gen_idx,
                    #                                                                             ampl_scale=self.post_proc_cfg.get('rel_ampl_scale', 2.0))
                    
                    obj_labels, obj_scores, obj_sample_prob = topk_score_selected_simple(all_pair_num, obj_scores, gen_idx=obj_gen_idx,
                                                                                  ampl_scale=self.post_proc_cfg.get('ent_ampl_scale', 4.0))
                    sub_labels, sub_scores, sub_sample_prob = topk_score_selected_simple(all_pair_num, sub_scores, gen_idx=sub_gen_idx,
                                                                                  ampl_scale=self.post_proc_cfg.get('ent_ampl_scale', 4.0)) 
                    predicate_labels, predicate_scores, predi_sample_prob = topk_score_selected_simple(all_predicate_num, predicate_scores,gen_idx=predicate_gen_idx,
                                                                                                       ampl_scale=self.post_proc_cfg.get('rel_ampl_scale', 2.0))



                    # obj_labels, obj_scores = sample_labels(all_pair_num, obj_scores)
                    # sub_labels, sub_scores = sample_labels(all_pair_num, sub_scores)
                    # predicate_labels, predicate_scores = sample_labels(all_predicate_num, predicate_scores)

                    if show_match_detail:
                        print("init label and score")
                        print(each_inst['sub']['label'], each_inst['obj']
                              ['label'], each_inst['predicate']['label'],)
                        print(sub_pred_scores, obj_pred_scores,
                              predicate_pred_scores)

                        def top_tkn_score(pred_dist):
                            scr, token = pred_dist.sort(
                                descending=True, dim=-1)
                            words = self.tokenizer.decode(
                                token[:, :10].reshape(-1))
                            scrs = scr[:, :10].reshape(-1)
                            return words, scrs

                        words, scrs = top_tkn_score(sub_pred_dist)
                        print(words)
                        print(scrs)
                        print([index(self.cate_dict['obj'], each)
                              for each in words.split(' ')])

                        words, scrs = top_tkn_score(obj_pred_dist)
                        print(words)
                        print(scrs)
                        print([index(self.cate_dict['obj'], each)
                              for each in words.split(' ')])

                        words, scrs = top_tkn_score(predicate_pred_dist)
                        print(words)
                        print(scrs)
                        print([index(self.cate_dict['rel'], each)
                              for each in words.split(' ')])

                        init_obj_cate_id = index(
                            self.cate_dict['obj'], each_inst['obj']['label'])
                        init_sub_cate_id = index(
                            self.cate_dict['obj'], each_inst['sub']['label'])
                        init_pred_cate_id = index(
                            self.cate_dict['rel'], each_inst['predicate']['label'])
                        print(init_sub_cate_id, init_obj_cate_id, init_pred_cate_id)

                        print()
                        print("sampled label and score:")
                        print(sub_labels, obj_labels, predicate_labels)
                        print(sub_sample_prob.sort(descending=True)[
                              0][:30], sub_sample_prob.sort(descending=True)[1][:30])
                        print(obj_sample_prob.sort(descending=True)[
                              0][:30], )
                        print(predi_sample_prob.sort(descending=True)[
                              0][:30], predi_sample_prob.sort(descending=True)[1][:30])

                        print([self.cate_dict['obj'][each.item()] for each in sub_labels])
                        print([ self.cate_dict['rel'][each.item() ] for each in predicate_labels])
                        print([ self.cate_dict['obj'][each.item()]  for each in obj_labels])

                        pred_idx = predi_sample_prob.sort(descending=True)[1][:30]
                        obj_pred_idx = obj_sample_prob.sort(descending=True)[1][:30]
                        sub_pred_idx = obj_sample_prob.sort(descending=True)[1][:30]

                        print([self.cate_dict['obj'][each.item()] for each in sub_pred_idx])
                        print([self.cate_dict['rel'][each.item() ] for each in pred_idx])
                        print([self.cate_dict['obj'][each.item()]  for each in obj_pred_idx])

                        if not sub_matched or not obj_matched or not pred_matched:
                            print(sub_gen_idx, init_sub_gen_idx, obj_gen_idx,
                                  init_obj_gen_idx, predicate_gen_idx, init_predicate_gen_idx)
                            print(idx)

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
                    # select_range = len(triplet_scores) // 2
                    # select_range_idx = torch.multinomial(predicate_scores_expanded, num_samples=select_range)

                    # trp_scrs = torch.stack(
                    #     (obj_scores_expanded, sub_scores_expanded, predicate_scores_expanded))

                    trp_scr, trp_idx = triplet_scores.sort(
                        dim=0, descending=True)
                    # trp_idx = trp_idx[: int(len(trp_idx)*0.8)]

                    # unexpand field initial index
                    init_indx = torch.ones(
                        all_pair_num**2 * all_predicate_num, dtype=int) * idx
                    init_inst_idx.append(init_indx)

                    # extend field
                    all_triplets_scores.append(triplet_scores)
                    all_obj_scores.append(obj_scores_expanded)
                    all_sub_scores.append(sub_scores_expanded)
                    all_predicates_scores.append(
                        predicate_scores_expanded)

                    all_pred_labels.append(predicate_labels_expanded)
                    all_obj_labels.append(obj_labels_expanded)
                    all_sub_labels.append(sub_labels_expanded)

                    # un-extend field
                    all_obj_dist.append(obj_scores)
                    all_sub_dist.append(sub_scores)
                    all_predicates_dist.append(predicate_scores)

                    all_obj_box.append(each_inst['obj']['boxes'])
                    all_sub_box.append(each_inst['sub']['boxes'])

                all_triplets_scores = torch.cat(all_triplets_scores).cpu()
                all_predicates_scores = torch.cat(all_predicates_scores).cpu()

                # sort by triplet score
                triplets_scores, triplets_indx = all_triplets_scores.sort(
                    dim=0, descending=True)
                all_predicates_scores_sorted = all_predicates_scores[triplets_indx]

                # sort by predicates score
                # all_predicates_scores_sorted, triplets_indx = all_predicates_scores.sort(
                #     dim=0, descending=True)
                # triplets_scores = all_triplets_scores[triplets_indx]

                # init order
                # triplets_scores = all_triplets_scores
                # triplets_indx = torch.arange(all_triplets_scores.shape[0])
                # all_predicates_scores_sorted = all_predicates_scores[triplets_indx]

                all_sub_scores_sorted = torch.cat(
                    all_sub_scores).cpu()[triplets_indx]
                all_obj_scores_sorted = torch.cat(
                    all_obj_scores).cpu()[triplets_indx]
                all_pred_labels_sorted = torch.cat(
                    all_pred_labels).cpu()[triplets_indx]
                all_obj_labels_sorted = torch.cat(
                    all_obj_labels).cpu()[triplets_indx]
                all_sub_labels_sorted = torch.cat(
                    all_sub_labels).cpu()[triplets_indx]

                # unexpand fields
                init_inst_idx_sorted = torch.cat(
                    init_inst_idx).cpu()[triplets_indx]
                all_obj_dist_sorted = torch.stack(
                    all_obj_dist).cpu()[init_inst_idx_sorted]
                all_sub_dist_sorted = torch.stack(
                    all_sub_dist).cpu()[init_inst_idx_sorted]
                all_predicates_dist_sorted = torch.stack(all_predicates_dist).cpu()[
                    init_inst_idx_sorted]

                all_obj_box_sorted = torch.stack(
                    all_obj_box).cpu()[init_inst_idx_sorted]
                all_sub_box_sorted = torch.stack(
                    all_sub_box).cpu()[init_inst_idx_sorted]

                # fill data format
                all_box = torch.cat(
                    (all_sub_box_sorted, all_obj_box_sorted), dim=0)
                all_ent_label = torch.cat(
                    (all_sub_labels_sorted, all_obj_labels_sorted))
                all_ent_score = torch.cat(
                    (all_sub_scores_sorted, all_obj_scores_sorted), dim=0)
                all_ent_dist = torch.cat(
                    (all_sub_dist_sorted, all_obj_dist_sorted), dim=0)

                all_box = xywh2xyxy(all_box)
                invalid_idx = all_box[:, 2:] < all_box[:, :2]
                all_box[:, :2][invalid_idx] = all_box[:,
                                                      2:][invalid_idx] - 0.001

                all_rel_tripelts = torch.stack((torch.arange(len(all_sub_labels_sorted)),
                                                torch.arange(len(all_sub_labels_sorted)) + len(all_sub_labels_sorted))).T

                if show_match_detail:
                    for t_id in trp_idx:
                        print(self.cate_dict['obj'][sub_labels_expanded[t_id].item()],
                              self.cate_dict['rel'][predicate_labels_expanded[t_id].item(
                              )],
                              self.cate_dict['obj'][obj_labels_expanded[t_id].item()], end=', ')
                    print()
                    print('==============')

                # Entity NMS
                start = time.perf_counter()

                iou_threshold = 0.9
                nms_keep_idx = box_ops.batched_nms(
                    all_box, all_ent_score, all_ent_label, iou_threshold=iou_threshold)

                all_ent_label_nms = all_ent_label[nms_keep_idx]
                all_ent_scores_nms = all_ent_score[nms_keep_idx]
                all_ent_box_nms = all_box[nms_keep_idx]
                all_ent_dist_nms = all_ent_dist[nms_keep_idx]

                new_rel_pair_idx = torch.zeros_like(all_rel_tripelts)
                for class_id in torch.unique(all_ent_label_nms):
                    nms_indices = torch.where(all_ent_label_nms == class_id)[0]
                    all_indices = torch.where(all_ent_label == class_id)[0]
                    match_iou = generalized_box_iou(
                        all_box[all_indices], all_ent_box_nms[nms_indices]) 
                    max_iou_val, match_idx = match_iou.max(-1) # all, nmsed_box_id
                    for m_i, select_i in enumerate(all_indices):
                        new_rel_pair_idx[all_rel_tripelts ==
                                         select_i] = nms_indices[match_idx[m_i]]

                all_rel_tripelts = new_rel_pair_idx
                all_ent_label = all_ent_label_nms
                all_ent_score = all_ent_scores_nms
                all_ent_dist = all_ent_dist_nms
                all_box = all_ent_box_nms

                # self connection remove
                self_iou = generalized_box_iou(all_box, all_box)
                sub_idx = all_rel_tripelts[:, 0]
                obj_idx = all_rel_tripelts[:, 1]

                boxes_pair_iou = self_iou[sub_idx, obj_idx]
                # self_conn_idx = torch.logical_and(boxes_pair_iou > 0.9,
                #                                       all_ent_label[sub_idx] == all_ent_label[obj_idx])
                self_conn_idx = boxes_pair_iou > 0.92
                non_self_conn_idx = torch.ones(len(all_rel_tripelts)).bool()
                non_self_conn_idx[self_conn_idx] = False

                all_rel_tripelts = all_rel_tripelts[non_self_conn_idx]
                all_pred_labels_sorted = all_pred_labels_sorted[non_self_conn_idx]
                triplets_scores = triplets_scores[non_self_conn_idx]
                all_predicates_scores_sorted = all_predicates_scores_sorted[non_self_conn_idx]
                all_predicates_dist_sorted = all_predicates_dist_sorted[non_self_conn_idx]

                # triplets NMS
                all_rel_tripelts, bin_mask = rel_prediction_filtering(
                    all_rel_tripelts, all_rel_tripelts[:, 0], all_rel_tripelts[:, 1],
                    all_ent_label, all_pred_labels_sorted, triplets_scores,
                    self_iou, overlap_thres=0.9
                )
                all_predicates_dist_sorted = all_predicates_dist_sorted[bin_mask]
                all_predicates_scores_sorted = all_predicates_scores_sorted[bin_mask]
                all_pred_labels_sorted = all_pred_labels_sorted[bin_mask]
                triplets_scores = triplets_scores[bin_mask]

                _, triplets_indx = triplets_scores.sort(
                    dim=0, descending=True)

                # print("rel_prediction_filtering", time.perf_counter()-start)
                # print(len(pred_rel_triplet_selected), len(bin_mask))

                prediction = BoxList(all_box,
                                     (img_orig_size[1], img_orig_size[0]), mode="xyxy")

                prediction.add_field('pred_labels', all_ent_label)
                prediction.add_field('pred_scores', all_ent_score)
                prediction.add_field('pred_dists', all_ent_dist)

                prediction.add_field(
                    'rel_pair_idxs', all_rel_tripelts[triplets_indx])  # N, 2
                prediction.add_field(
                    'pred_rel_dist', all_predicates_dist_sorted[triplets_indx])
                prediction.add_field(
                    'pred_rel_score', all_predicates_scores_sorted[triplets_indx])
                prediction.add_field(
                    'pred_rel_label', all_pred_labels_sorted[triplets_indx])
                prediction.add_field('pred_rel_trp_score',
                                     triplets_scores[triplets_indx])

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
        max_txt_len = cfg.get("max_txt_len", 512)
        logger.info("model CFG for load")
        logger.info(cfg)

        model = cls(image_encoder, text_decoder, prompt=prompt, add_noise=cfg.get("add_noise", False),
                    max_txt_len=max_txt_len,
                    dump_pred=cfg.get("dump_pred", False),
                    reduction=cfg.get("reduction", 'mean'),
                    num_coord_bin=cfg.get("num_coord_bin", 1000),
                    mask_label_ratio=cfg.get("mask_label_ratio", 0.0),
                    top_k_label_num=cfg.get("top_k_ent_label_num", 3),
                    top_k_predicate_label_num=cfg.get(
                        "top_k_predicate_label_num", 3),
                    aux_close_classifier=cfg.get(
                        "aux_close_classifier", False),
                    cate_dict_url=cfg.get("cate_dict_url", ""),
                    dump_dir=cfg.get("dump_dir", None),
                    box_loss_weight=cfg.get("box_loss_weight", 1.0),
                    seg_len=cfg.get("seg_len", 64),
                    post_proc_cfg=cfg.get("post_proc_cfg", None),)



        med_config_path = get_abs_path(cfg.get("med_config_path"))
        med_config = BertConfig.from_json_file(med_config_path)

        vocab_size_init = med_config.vocab_size

        med_config.vocab_size += SPEICAL_TOKEN_NUM
        logger.info(
            f"med_config.vocab_size {vocab_size_init} + SPE_TKN:{SPEICAL_TOKEN_NUM} -> {med_config.vocab_size}")
        logger.info("med_config")
        logger.info(str(med_config))
        msg_missing_keys = [1,]
        try:
            msg = model.load_checkpoint_from_config(cfg)
            msg_missing_keys = msg.missing_keys
        except :
            pass
        # from pretrain
        if len(msg_missing_keys) > 0:
            text_decoder.resize_token_embeddings(med_config.vocab_size)
            model.load_checkpoint_from_config(cfg)

        text_decoder.resize_token_embeddings(med_config.vocab_size)

        return model





class XBertLMHeadDecoderDetHead(XBertLMHeadDecoder):

    def __init__(self, config, pos_adapter=False, pos_adapter_tfmer_layer=6, conv_module='none'):
        super(XBertLMHeadDecoder, self).__init__(config)

        self.init_weights()
        hidden_dim_in = 768
        hidden_dim = 256
        self.pos_adapter_on = pos_adapter
        if self.pos_adapter_on:

            self.conv_module = None
            if conv_module == 'light_conv':
                self.conv_module = LightweightConv(hidden_dim)
            elif conv_module == 'res18':
                self.res18 = Res18Wrapper(out_channels=hidden_dim)
                self.conv_module = self.res18

            self.ent_hs_input_proj = nn.Linear(hidden_dim_in, hidden_dim)

            self.pos_encoder = None
            if pos_adapter_tfmer_layer > 0:
                self.position_embedding = PositionEmbeddingSine(
                    num_pos_feats=hidden_dim // 2,
                    temperature=10000,
                    normalize=True,
                    scale=None,)
                self.enc_input_proj = nn.Linear(hidden_dim_in, hidden_dim)
                self.pos_encoder = build_encoder(
                    num_decoder_layers=pos_adapter_tfmer_layer, d_model=hidden_dim)
                self.pos_decoder = build_decoder(
                    num_decoder_layers=pos_adapter_tfmer_layer, d_model=hidden_dim,
                    return_intermediate_dec=True)

            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    @classmethod
    def from_config(cls, cfg, from_pretrained=False):

        med_config_path = get_abs_path(cfg.get("med_config_path"))
        med_config = BertConfig.from_json_file(med_config_path)

        if from_pretrained:
            print("load from pretrained bert-base-uncased")
            return cls.from_pretrained("/public/home/lirj2/projects/LAVIS_GITM/data/bert-base-uncased",
                                        config=med_config, ignore_mismatched_sizes=False)
        else:
            return cls(config=med_config, pos_adapter=cfg.get("pos_adapter", False),
                       conv_module=cfg.get("pos_adapter_conv", 'none'),
                       pos_adapter_tfmer_layer=cfg.get("pos_adapter_tfmer_layer", 0))

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

        ent_hs_padded = self.ent_hs_input_proj(ent_hs_padded)

        if self.pos_encoder is not None:
            vis_enc_hs = encoder_hidden_states
            vis_enc_hs = self.enc_input_proj(vis_enc_hs)

            feat_w = int(vis_enc_hs[:, 1:].shape[1] ** 0.5)
            vis_enc_map = vis_enc_hs[:, 1:].transpose(1, 0)
            vis_enc_mask = encoder_attention_mask[:, 1:].reshape(
                vis_enc_hs.shape[0], feat_w, feat_w,)
            vis_enc_mask = (1 - vis_enc_mask).bool()
            # position coding
            vis_enc_pos_emb = self.position_embedding(
                vis_enc_map, vis_enc_mask).flatten(2).permute(2, 0, 1)
            vis_enc_mask = vis_enc_mask.flatten(1)

            # conv_out = self.conv_module(raw_image)
            if self.conv_module is not None:
                conv_out = self.conv_module(raw_image)
                vis_enc_map = vis_enc_map + conv_out.flatten(-2).permute(2, 0, 1)

            encoder_hs = self.pos_encoder(  # T, B, D; B, T
                src=vis_enc_map, src_key_padding_mask=vis_enc_mask, pos=vis_enc_pos_emb
            )  # -> T, B, D
            # query decodeing
            ent_hs_padded = ent_hs_padded.transpose(1, 0)
            decode_ent_hs = self.pos_decoder(
                tgt=ent_hs_padded.to(device),
                memory=encoder_hs.to(device),
                tgt_key_padding_mask=ent_padding_mask.to(device),
                memory_key_padding_mask=vis_enc_mask.to(device),
                pos=vis_enc_pos_emb.to(device)
            )# lyr, T, B, D
        else:
            # directly predict the position
            decode_ent_hs = ent_hs_padded.transpose(1, 0)
        # pos prediction
        ent_box_pred = self.bbox_embed(
            decode_ent_hs).sigmoid() # lyr, T, B, D
        ent_box_pred_out = ent_box_pred[-1].transpose(1, 0) 
        pred_extracted_box = []
        extracted_box_seg = [len(each) for each in entity_hs]

        for bid, each in enumerate(entity_hs):
            pred_extracted_box.append(ent_box_pred_out[bid][:len(each)])
        # output packing
        outputs = {'extracted_box': pred_extracted_box,
                   'extracted_box_seg': extracted_box_seg}
        
        # loss calculation
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
            bbox_giou_loss_lyr = []
            bbox_giou_lyr = []
            bbox_l1_loss_lyr = []
            for lyr in range(len(ent_box_pred)):
                ent_box_pred_out = ent_box_pred[lyr].transpose(1, 0) 
                lyr_pred_extracted_box = []
                extracted_box_seg = [len(each) for each in entity_hs]
                for bid, each in enumerate(entity_hs):
                    lyr_pred_extracted_box.append(ent_box_pred_out[bid][:len(each)])

                lyr_pred_box_cat = torch.cat(lyr_pred_extracted_box)
                bbox_l1 = F.l1_loss(
                    lyr_pred_box_cat, gt_box_cat_all, reduction='none').sum(-1).mean()

                bbox_giou = torch.diag(generalized_box_iou(
                    box_cxcywh_to_xyxy(lyr_pred_box_cat),
                    box_cxcywh_to_xyxy(gt_box_cat_all)
                ))

                bbox_giou_loss = (1 - bbox_giou).mean()

                bbox_giou_loss_lyr.append(bbox_giou_loss)
                bbox_giou_lyr.append(bbox_giou.mean())
                bbox_l1_loss_lyr.append(bbox_l1)

            bbox_giou_loss_lyr = torch.stack(bbox_giou_loss_lyr)
            bbox_l1_loss_lyr = torch.stack(bbox_l1_loss_lyr)
            bbox_giou_lyr = torch.stack(bbox_giou_lyr)

            outputs['pos_adp_loss'] = {
                'pos_adp_total_loss': torch.sum(bbox_l1_loss_lyr[:-1]) + bbox_l1_loss_lyr[-1] * 2 \
                                      + torch.sum(bbox_giou_loss_lyr[:-1]) + bbox_giou_loss_lyr[-1] * 5,
                'pos_adp_bbox_giou': torch.mean(bbox_giou_lyr),
                'pos_adp_bbox_l1': torch.mean(bbox_l1_loss_lyr)
            }
            lyr_loss  = {}
            for lyr_i in range(len(bbox_l1_loss_lyr)):
                lyr_loss.update({
                    f"pos_adp_bbox_giou_lyr-{lyr_i}": bbox_giou_lyr[lyr_i],
                    f"pos_adp_bbox_l1_lyr-{lyr_i}": bbox_l1_loss_lyr[lyr_i],
                })
            outputs['pos_adp_loss'].update(lyr_loss)
            
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

            # build box loss weights
            box_l1 = []
            box_giou = []
            box_loss_weights = torch.ones_like(labels).float()
            pos_ada_output = None
            if box_targets is not None:
                box_loss_weights = torch.ones_like(labels) * 1.0

                if self.pos_adapter_on:
                    pred_ent_hs_all = []
                    for bid, box_gts in enumerate(box_targets):
                        pred_ent_hs = []
                        pred_box_seq = torch.argmax(
                            shifted_prediction_scores[bid].contiguous(), dim=-1)
                        for bgt in box_gts:
                            tkn_pos = bgt['tkn_pos']
                            # xywh
                            pred_ent_hs.append(
                                sequence_output[bid, tkn_pos-1] + sequence_output[bid, tkn_pos-2])  # obj token + categories token
                        pred_ent_hs = torch.stack(pred_ent_hs)
                        pred_ent_hs_all.append(pred_ent_hs)

                    pos_ada_output = self.pos_adapter(
                        pred_ent_hs_all, raw_image, encoder_hidden_states, encoder_attention_mask, box_targets)

            box_loss_weights[token_type == COORD] *= box_loss_weights_all
            label_smoothing = 0.1
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
                notif_loss = (
                    loss_none_box_weighted[token_type == NOTIF].sum() / len(torch.nonzero(token_type == NOTIF))).mean()

                word_acc = sum(shifted_prediction_scores[token_type == WORD].max(-1)[
                               1] == labels[token_type == WORD]) / labels[token_type == WORD].view(-1).shape[0]
                notif_acc = sum(shifted_prediction_scores[token_type == NOTIF].max(-1)[
                                1] == labels[token_type == NOTIF]) / labels[token_type == NOTIF].view(-1).shape[0]

            word_loss = word_loss.mean()
            # coord_loss = coord_loss.mean()
            notif_loss = notif_loss.mean()

            box_l1 = torch.Tensor(box_l1)
            box_giou = torch.Tensor(box_giou)
            loss_dict = {
                'loss': lm_loss,
                'init_lm_loss': lm_loss_wo_box,
                'word_loss': word_loss,
                # "coord_loss": coord_loss,
                "notif_loss": notif_loss,
                "notif_acc": notif_acc,
                "word_acc": word_acc,
                # 'coord_error': coord_error,
                # 'box_l1_error': torch.mean(box_l1),
                # 'box_giou': torch.mean(box_giou)
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
            num_beams = 1
            visual_embeds = visual_embeds

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


def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor


def index(seqs, tar):
    try:
        idx = seqs.index(tar)
        return idx
    except:
        return -1


def rel_prediction_filtering(all_rel_tripelts, sub_idx, obj_idx,
                             ent_label, rel_pred_label, rel_pred_score,
                             self_iou, overlap_thres=0.8):
    """

    Args:
        pred_idx_set:
        new_come_pred_idx:

    Returns:

    """
    pred_idx_set = []
    for new_come_pred_idx in range(len(all_rel_tripelts)):

        new_come_sub_idx = sub_idx[new_come_pred_idx]
        new_come_obj_idx = obj_idx[new_come_pred_idx]

        new_come_sub_label = ent_label[new_come_sub_idx]
        new_come_obj_label = ent_label[new_come_obj_idx]

        new_come_pred_label = rel_pred_label[new_come_pred_idx]
        new_come_pred_score = rel_pred_score[new_come_pred_idx]

        pred_idx = torch.Tensor(pred_idx_set).long()
        curr_sub_idx = sub_idx[pred_idx]
        curr_obj_idx = obj_idx[pred_idx]

        curr_sub_label = ent_label[curr_sub_idx]
        curr_obj_label = ent_label[curr_obj_idx]

        curr_pred_label = rel_pred_label[pred_idx]
        curr_pred_score = rel_pred_score[pred_idx]

        entities_indx_match = torch.logical_and(
            curr_sub_idx == new_come_sub_idx,
            curr_obj_idx == new_come_obj_idx
        )

        new_come_sub_idx = (torch.ones(len(pred_idx))
                            * new_come_sub_idx).long()
        new_come_obj_idx = (torch.ones(len(pred_idx))
                            * new_come_obj_idx).long()

        sub_iou = self_iou[new_come_sub_idx, curr_sub_idx]
        obj_iou = self_iou[new_come_obj_idx, curr_obj_idx]

        entities_pred_match = torch.logical_and(
            torch.logical_and(sub_iou > overlap_thres,
                              obj_iou > overlap_thres),
            torch.logical_and(curr_sub_label == new_come_sub_label,
                              curr_obj_label == new_come_obj_label)
        )
        entity_match = torch.logical_or(
            entities_pred_match, entities_indx_match)

        if entity_match.any():
            pred_match = curr_pred_label == new_come_pred_label
            rel_match = torch.logical_and(
                entity_match, pred_match)  # pred_idx_len

            if rel_match.any():
                rel_match_idx = squeeze_tensor(torch.nonzero(rel_match))
                is_existed = curr_pred_score[rel_match] < new_come_pred_score
                if is_existed.any():
                    # add higher score prediction idx and remove the lower score prediction
                    pred_idx_set.append(new_come_pred_idx)
                    for repeat_idx in rel_match_idx[is_existed].tolist():
                        pred_idx_set.pop(repeat_idx)
            else:
                pred_idx_set.append(new_come_pred_idx)

        else:
            pred_idx_set.append(new_come_pred_idx)
    device = all_rel_tripelts.device
    pred_idx_set = torch.unique(torch.Tensor(pred_idx_set).long()).to(device)
    bin_mask = torch.zeros((all_rel_tripelts.shape[0]), dtype=torch.bool).to(
        device
    )
    bin_mask[pred_idx_set] = True
    pred_rel_triplet_selected = all_rel_tripelts[bin_mask]

    return pred_rel_triplet_selected, bin_mask


class LogitsWarper:
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, top_p=0.8, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.top_k = top_k
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, scores: torch.FloatTensor, keep_idx=None) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep),
                    scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[
            0][..., -1, None]
        if keep_idx is not None:
            indices_to_remove[..., keep_idx] = False
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores
