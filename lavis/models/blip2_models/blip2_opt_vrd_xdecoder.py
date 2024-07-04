"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from ast import List
import copy
import json
import logging
import os
import pickle
import random
from typing import List, Optional, Tuple, Union
import contextlib

import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from torchvision.ops import boxes as box_ops
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


from lavis.common.registry import registry
from lavis.datasets.datasets.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import OPTForCausalLM, OPTConfig
# from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from lavis.models.blip_models.blip_det import COORD, NOTIF, WORD, check_label_name, find_token
from lavis.models.blip_models.blip_rel_det_pgsg import index, rel_prediction_filtering
from lavis.models.detr_transformer import MLP, PositionEmbeddingSine, build_decoder, build_encoder

from lavis.models.resnet import LightweightConv, Res18Wrapper
from lavis.models.weight_init import show_params_status
from lavis.tasks.evaluation.boxlist import BoxList


logger = logging.getLogger(__name__)


def get_token_type(labels):
    token_types = torch.zeros_like(labels)
    token_types[labels < 50265] = WORD
    token_types[labels < 0] = 0
    token_types[labels >= 50265] = NOTIF
    return token_types


@registry.register_model("blip2_opt_vrd")
class Blip2OPTVrd(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_psg": "configs/models/blip2/blip2_psg_vrd_opt2.7b.yaml",
        "base_oiv6": "configs/models/blip2/blip2_oiv6_vrd_opt2.7b.yaml",

        # "pretrain_opt2.7b": "lavis/configs/models/blip2/blip2_vrd_opt2.7b.yaml",
        # "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        # "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def maybe_autocast(self, dtype=torch.float16):
        # disable cast to fp16
        return contextlib.nullcontext()

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        max_objects=99,
        max_pos_objects=20, num_coord_bin=1000,
        dump_pred=False, reduction='none', top_k_label_num=5, top_k_predicate_label_num=3,
        mask_label_ratio=0.5, cate_dict_url="", box_loss_weight=1.,
        dump_dir='/mnt/petrelfs/lirongjie/project/LAVIS/lavis/output/BLIP/vis_dump',
        pos_adapter=True,
        pos_adapter_conv="none",
        pos_adapter_tfmer_layer=6,
        finetune_strategy='lora',  # partial_lyr, lora
        finetune_layer_num=4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "k_proj",
            "q_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2"
        ]

    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_model = OPTLMHeadDecoderDetHead.from_pretrained(
            opt_model,
            # torch_dtype=torch.float16,
            torch_dtype=torch.float32,
            pos_adapter=pos_adapter, pos_adapter_conv=pos_adapter_conv,
            pos_adapter_tfmer_layer=pos_adapter_tfmer_layer,
            vis_in_dim=self.visual_encoder.num_features,
            local_files_only=True,
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False

        
        if finetune_strategy == 'lora':
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.opt_model.model = get_peft_model(self.opt_model.model, lora_config)

        elif 'partial_lyr' in finetune_strategy:
            if 'lora' in finetune_strategy:
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="none",
                )
                        
                for layer_id in range(len(self.opt_model.base_model.decoder.layers)-finetune_layer_num,
                                    len(self.opt_model.base_model.decoder.layers)):
                    self.opt_model.base_model.decoder.layers[layer_id] = get_peft_model(self.opt_model.base_model.decoder.layers[layer_id], lora_config)
            else:
                for layer_id in range(len(self.opt_model.base_model.decoder.layers)-finetune_layer_num,
                                    len(self.opt_model.base_model.decoder.layers)):
                    for name, param in self.opt_model.base_model.decoder.layers[layer_id].named_parameters():
                        param.requires_grad = True

        for name, param in self.opt_model.named_parameters():
            if 'pos_adapter' in name:
                param.requires_grad = True
            elif 'lm_head' in name:
                param.requires_grad = True
            elif 'embed_tokens' in name:
                param.requires_grad = True

        self.opt_model.lm_head.weight.requires_grad = True
        self.opt_model.model.decoder.embed_tokens.weight.requires_grad = True

        # initialize tokenizer
        self.tokenizer = StagedGPT2Tokenizer.from_pretrained(
            opt_model, use_fast=False, add_prefix_space=False, local_files_only=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, add_prefix_space=True)

        self.text_vocab_size = len(self.tokenizer)
        self.tokenizer.add_tokens(["[OBJ]"])
        self.tokenizer.add_tokens(["[REL]"])
        self.ent_token_id = self.tokenizer(
            "[OBJ]", add_special_tokens=False).input_ids[0]
        self.rel_token_id = self.tokenizer(
            "[REL]", add_special_tokens=False).input_ids[0]
        self.eos_token_id = self.tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self.max_objects = max_objects  # all object for training
        self.max_pos_objects = max_pos_objects  # the number of postive sample

        self.num_coord_bin = num_coord_bin

        self.top_k_label_num = top_k_label_num
        self.top_k_predicate_label_num = top_k_predicate_label_num
        self.mask_label_ratio = mask_label_ratio
        self.box_loss_weight = box_loss_weight

        with open(cate_dict_url, 'r') as f:
            self.cate_dict = json.load(f)

        self.vocab2category = DynamicClassifier(
            tokenizer=self.tokenizer, cate_dict=self.cate_dict,
            word_embedding_predictor=self.opt_model.lm_head,
            word_embedding_predictor_weight=self.opt_model.lm_head.weight
        )
        self.dump_pred = dump_pred
        self.dump_dir = dump_dir

        self.seq2instance = Sequence2Instance(
            self.prompt_length, self.cate_dict, self.tokenizer, top_k_label_num, top_k_predicate_label_num
        )

        logger.info(show_params_status(self))

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )  # batch, query, hid
        outputs = self.forward_decoder(
            samples, image_embeds, query_output, image_atts)
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():

            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(
                image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.tokenizer(prompt, return_tensors="pt").to(
                image.device
            )
            attention_mask = torch.cat(
                [atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                num_beams = 1

            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

            decoder_out_collect = None
            seg_len = 32
            min_seq_len = 16
            for i in range(max_length // seg_len):
                decoder_out = self.opt_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=seg_len,
                    min_length=min_seq_len,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_scores=True,
                )
                trim_hs = []
                for gen_step, new_hs_each in enumerate(decoder_out.hidden_states):
                    trim_hs.append(new_hs_each[-1][:, -1:][None, :])


                decoder_out.hidden_states = tuple(trim_hs)
                prompt_length = 1
                if decoder_out_collect is None:
                    decoder_out_collect = decoder_out
                else:

                    sequences_wo_propmt = decoder_out.sequences[:,
                                                                prompt_length:]
                    decoder_out_collect.sequences = torch.cat(
                        (decoder_out_collect.sequences, sequences_wo_propmt), dim=-1)

                    decoder_out_collect.scores = decoder_out_collect.scores + decoder_out.scores
                    decoder_out_collect.hidden_states = decoder_out_collect.hidden_states + \
                        decoder_out.hidden_states

        decoder_out = decoder_out_collect
        self.prompt_length = opt_tokens.input_ids.shape[1]

        raw_caption = [self.tokenizer.decode(seq, skip_special_tokens=False, decode_raw_tokens=True)
                       for seq in decoder_out.sequences]
        
        batch_object_list = self.seq2instance(decoder_out_collect, raw_caption,
                                              prompt_length=prompt_length,
                                              decoder_hidden_states=decoder_out_collect.hidden_states,
                                              verbose=False)

        batch_object_list_cate_trans = self.vocab2category(
            batch_object_list)

        if self.opt_model.pos_adapter_on:
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
                # if len(ent_hs) <= 0:
                #     import ipdb; ipdb.set_trace()
                pred_ent_hs_all.append(torch.stack(ent_hs))

            pos_ada_output = self.opt_model.pos_adapter(
                pred_ent_hs_all, samples['image'], image_embeds, encoder_attention_mask, None)

            role_list = ['sub', 'obj']
            for bid, adapter_box_pred in enumerate(pos_ada_output['extracted_box']):
                for inst_id, each_box_pred in enumerate(adapter_box_pred):
                    batch_object_list_cate_trans[bid][int(
                        inst_id//2)][role_list[inst_id % 2]]['boxes'] = each_box_pred

        predictions, ground_truths, image_info = self.seq2instance.postprocess(
            batch_object_list_cate_trans, samples['targets'])

        if self.dump_pred:
            (forward_decoder_output,
             decoder_targets,
             box_targets) = self.forward_decoder(samples, image_embeds, query_output, image_atts, return_box_gts_batch=True)

            raw_caption_target = [self.tokenizer.decode(decoder_targets[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                  for i in range(len(decoder_targets))]
            f_dec_tokens = torch.argmax(
                forward_decoder_output.logits.contiguous(), dim=-1)
            raw_caption_fdec = [self.tokenizer.decode(f_dec_tokens[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                for i in range(len(f_dec_tokens))]

            pos_ada_output = None
            if self.opt_model.pos_adapter_on:
                pred_ent_hs_all = []
                sequence_output = forward_decoder_output.hidden_states[-1]

                for bid, box_gts in enumerate(box_targets):
                    pred_ent_hs = []
                    for bgt in box_gts:
                        tkn_pos = bgt['tkn_pos']
                        # xywh
                        pred_ent_hs.append(
                            sequence_output[bid, tkn_pos-1] + sequence_output[bid, tkn_pos-2])  # obj token + categories token
                    pred_ent_hs = torch.stack(pred_ent_hs)
                    pred_ent_hs_all.append(pred_ent_hs)

                pos_ada_output = self.opt_model.pos_adapter(
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

    def forward_decoder(self, samples, image_embeds, query_output, image_atts, return_box_gts_batch=False):
        # prepare inputs for forwarding decoder
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(
            inputs_opt.size()[:-1], dtype=torch.long).to(image_embeds.device)

        self.tokenizer.padding_side = "right"

        # target_tokens

        if samples.get('targets') is not None:
            input_token, targets_tokens, close_vocab_classes, box_gts_batch = self.target2token_seqs(
                samples["targets"], text_mask_ratio=0.0)

            if self.prompt:
                targets_tokens[:, : self.prompt_length] = -100
                # do not apply loss to the prompt

        # input tokens
        inputs_embeds = self.opt_model.model.decoder.embed_tokens(
            input_token.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_opt, input_token.attention_mask], dim=1)

        # print("inputs_embeds", torch.isnan(inputs_embeds).nonzero())
        # print("image_embeds", torch.isnan(image_embeds).nonzero())

        with self.maybe_autocast():
            decoder_output = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                labels=targets_tokens,
                box_targets=box_gts_batch,
            )

        if return_box_gts_batch:
            return decoder_output, targets_tokens, box_gts_batch

        return decoder_output

    def target2token_seqs(self, targets, text_mask_ratio=0.5):

        tar_seqs = []
        # construct the templets
        for b_i, target in enumerate(targets):
            # dict_keys(['boxes', 'det_labels', 'rel_tripets', 'image_id', 'orig_size', 'size', 'det_labels_texts', 'rel_labels_texts'])
            all_idx = torch.randperm(len(target['rel_tripets'])).tolist()
            if len(all_idx) > self.max_objects:
                all_idx = all_idx[:self.max_objects]

            target['rel_tripets'] = target['rel_tripets'][all_idx]
            det_label = target['det_labels']
            rel_tri_text = []
            for each_trp in target['rel_tripets']:
                trp_idx = each_trp.tolist()
                # rel_tri_text.append(
                #     f"{self.cate_dict['obj'][det_label[trp_idx[0]]]} {self.cate_dict['rel'][trp_idx[2]]} {self.cate_dict['obj'][det_label[trp_idx[1]]]} ")

                rel_tri_text.append(
                    f"{self.cate_dict['obj'][det_label[trp_idx[0]]]} [OBJ] {self.cate_dict['rel'][trp_idx[2]]} [REL] {self.cate_dict['obj'][det_label[trp_idx[1]]]} [OBJ]")

            tar_seq = self.prompt + " "
            for idx, each_rel_txt in enumerate(rel_tri_text):
                tar_seq += each_rel_txt
                if idx < len(rel_tri_text) - 1:
                    if random.random() > 0.5:
                        tar_seq += ' , '
                    else:
                        tar_seq += ' and '
            tar_seq += '.\n'
            tar_seqs.append(tar_seq)

        label_token = self.tokenizer(
            tar_seqs,
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        raw_token = []
        close_vocab_classes = []
        box_gts_batch = []
        for b_i, target in enumerate(targets):
            boxes = target['boxes']
            boxes_xywh = copy.deepcopy(boxes)
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
                    if label_token_fuse[rel_idx] == self.ent_token_id:
                        if coord_id < 4:
                            box_gt.append({
                                'xywh': boxes_xywh[each_trp[0]],
                                'tkn_pos': rel_idx
                            })
                        else:
                            box_gt.append({
                                'xywh': boxes_xywh[each_trp[1]],
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

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_objects=cfg.get("max_objects", 16),
            pos_adapter=cfg.get("pos_adapter", False),
            pos_adapter_conv=cfg.get("pos_adapter_conv", 'none'),
            pos_adapter_tfmer_layer=cfg.get("pos_adapter_tfmer_layer", 0),
            cate_dict_url=cfg.get("cate_dict_url", 0),
            dump_pred=cfg.get("dump_pred", False),
            dump_dir=cfg.get("dump_dir", 0),
            top_k_label_num=cfg.get("top_k_ent_label_num", 3),
            top_k_predicate_label_num=cfg.get(
                "top_k_predicate_label_num", 3),
            finetune_strategy=cfg.get('finetune_strategy', 'partial_lyr'),
            finetune_layer_num=cfg.get('finetune_layer_num', 3)
        )


        # [50264, 50265, 50266]
        # ['<mask>', '[OBJ]', '[REL]']
        if not cfg.load_finetuned:
            model.opt_model.resize_token_embeddings(len(model.tokenizer))
        msg = model.load_checkpoint_from_config(cfg)
            # from pretrain
        if len(msg.missing_keys) > 0:
            model.opt_model.resize_token_embeddings(len(model.tokenizer))
            model.load_checkpoint_from_config(cfg)

        model.opt_model.resize_token_embeddings(len(model.tokenizer))
        
        return model


class OPTLMHeadDecoderDetHead(OPTForCausalLM):

    def __init__(self, config, pos_adapter=False, vis_in_dim=768,
                 pos_adapter_conv='none', pos_adapter_tfmer_layer=0,
                 label_smoothing = 0.00):
        super(OPTLMHeadDecoderDetHead, self).__init__(config)
        self.pos_adapter_on = pos_adapter
        self.label_smoothing = label_smoothing
        if self.pos_adapter_on:
            #
            self.pos_adapter = PosAdapter(pos_adapter_conv, pos_adapter_tfmer_layer,
                                          lang_in_dim=config.hidden_size,
                                          vis_in_dim=vis_in_dim)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict: Optional[bool] = None,
        reduction: Optional[str] = "mean",
        soft_labels=None,
        alpha=0,
        raw_image=None,
        box_targets=None,
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs[0]

        prediction_scores = self.lm_head(outputs[0]).contiguous()

        # print("outputs[0]", torch.isnan(outputs[0]).nonzero())
        # print("prediction_scores", torch.isnan(prediction_scores).nonzero())

        lm_loss = None
        loss_dict = {}
        if labels is not None:
            prediction_scores = prediction_scores[:, -labels.size(1):, :]
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:,
                                                          :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            token_type = get_token_type(labels)

            # build box loss weights
            box_l1 = []
            box_giou = []
            pos_ada_output = None
            if box_targets is not None:

                if self.pos_adapter_on:

                    sequence_output_trim = sequence_output[:, -
                                                           labels.size(1) - 1:]
                    pred_ent_hs_all = []
                    for bid, box_gts in enumerate(box_targets):
                        pred_ent_hs = []
                        pred_box_seq = torch.argmax(
                            shifted_prediction_scores[bid].contiguous(), dim=-1)
                        for bgt in box_gts:
                            tkn_pos = bgt['tkn_pos']
                            # xywh
                            pred_ent_hs.append(
                                sequence_output_trim[bid, tkn_pos-1] + sequence_output_trim[bid, tkn_pos-2])  # obj token + categories token
                        pred_ent_hs = torch.stack(pred_ent_hs)
                        pred_ent_hs_all.append(pred_ent_hs)

                    pos_ada_output = self.pos_adapter(
                        pred_ent_hs_all, raw_image, encoder_hidden_states, encoder_attention_mask, box_targets)

            label_smoothing = self.label_smoothing
            if reduction == "mean":
                loss_fct = torch.nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing, reduction="none", ignore_index=-100)
                init_lm_loss = loss_fct(
                    shifted_prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )
                weights = torch.ones(init_lm_loss.shape).to(
                    init_lm_loss.device)
                weights[labels.view(-1) == -100] = 0.00

                lm_loss = init_lm_loss * weights

                valid_inst_num = len(torch.nonzero(weights.view(-1) > 0.01))

                lm_loss = lm_loss.sum() / (valid_inst_num + 0.001)

                loss_none = loss_fct(shifted_prediction_scores.permute(
                    0, 2, 1), labels)  # batch_size, seq
                word_loss = torch.mean(
                    loss_none[token_type == WORD].sum() / (len(torch.nonzero(token_type == WORD)) + 0.001))
                notif_loss = torch.mean(
                    loss_none[token_type == NOTIF].sum() / (len(torch.nonzero(token_type == NOTIF)) + 0.001))

                word_acc = sum(shifted_prediction_scores[token_type == WORD].max(-1)[
                               1] == labels[token_type == WORD]) / (labels[token_type == WORD].view(-1).shape[0] + 0.001)
                notif_acc = sum(shifted_prediction_scores[token_type == NOTIF].max(-1)[
                                1] == labels[token_type == NOTIF]) / (labels[token_type == NOTIF].view(-1).shape[0] + 0.001)

            word_loss = word_loss.mean()
            # coord_loss = coord_loss.mean()
            notif_loss = notif_loss.mean()

            box_l1 = torch.Tensor(box_l1)
            box_giou = torch.Tensor(box_giou)
            loss_dict = {
                'loss': lm_loss,
                'word_loss': word_loss,
                "notif_loss": notif_loss,
                "notif_acc": notif_acc,
                "word_acc": word_acc,
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

        # print(torch.abs(self.lm_head.weight -
        #       self.model.decoder.embed_tokens.weight))

        return CausalLMOutputWithPast(
            loss=loss_dict,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # cross_attentions=outputs.cross_attentions,
        )


class StagedGPT2Tokenizer(GPT2Tokenizer):

    def _decode(
        self,
        token_ids: list,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop(
            "use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)

        if kwargs.pop("decode_raw_tokens", False):
            tokens = self.convert_ids_to_tokens(token_ids)
            tokens = [each if each is not None else 'None' for each in tokens]
            return tokens

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(
                        self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def convert_tokens_to_string(self, tokens):
        texts = "".join(tokens)
        # text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
    
        decode_collects = []
        for txt in texts:
            dec_res = self.byte_decoder.get(txt)
            if dec_res is None:
                decode_collects.append(' ')
            else:
                decode_collects.append(dec_res)
        output_text = bytearray(decode_collects).decode("utf-8", errors=self.errors)
        return output_text
    
    def clean_text_from_decode(self, filtered_tokens, spaces_between_special_tokens=True, clean_up_tokenization_spaces=True, skip_special_tokens=False):
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(
                        self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text


class PosAdapter(nn.Module):

    def __init__(self, conv_module='none', pos_adapter_tfmer_layer=0,
                 lang_in_dim=768, vis_in_dim=768, hidden_dim=256) -> None:
        super(PosAdapter, self).__init__()

        self.conv_module = None
        if conv_module == 'light_conv':
            self.conv_module = LightweightConv(hidden_dim)
        elif conv_module == 'res18':
            self.res18 = Res18Wrapper(out_channels=hidden_dim)
            self.conv_module = self.res18

        self.ent_hs_input_proj = nn.Linear(lang_in_dim, hidden_dim)

        self.pos_encoder = None
        if pos_adapter_tfmer_layer > 0:
            self.position_embedding = PositionEmbeddingSine(
                num_pos_feats=hidden_dim // 2,
                temperature=10000,
                normalize=True,
                scale=None,)
            self.enc_input_proj = nn.Linear(vis_in_dim, hidden_dim)
            self.pos_encoder = build_encoder(
                num_decoder_layers=pos_adapter_tfmer_layer, d_model=hidden_dim)
            self.pos_decoder = build_decoder(
                num_decoder_layers=pos_adapter_tfmer_layer, d_model=hidden_dim,
                return_intermediate_dec=True)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, entity_hs, raw_image, encoder_hidden_states, encoder_attention_mask, box_targets=None):
        """
        encoder_hidden_states: B T D
        encoder_attention_mask:B T D

        """
        # use selected token as query for detect things from the encoder features
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
        # todo vis encoder output may not satisfy the reshape pipeline
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
                vis_enc_map = vis_enc_map + \
                    conv_out.flatten(-2).permute(2, 0, 1)

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
            )

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


#  no learnable parameter
class DynamicClassifier:
    def __init__(self, tokenizer, cate_dict, word_embedding_predictor, word_embedding_predictor_weight):
        self.tokenizer = tokenizer
        self.cate_dict = cate_dict
        # self.word_embedding_weight = self.text_decoder.cls.predictions.decoder.weight.data
        self.word_embedding_predictor = word_embedding_predictor
        self.word_embedding_weight = word_embedding_predictor_weight

        self.init_category_space()

    def init_category_space(self):

        rel_cates = [' ' + each for each in self.cate_dict['rel']]
        self.rel_label_token = self.tokenizer(
            rel_cates,
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).input_ids[:, 1:]

        obj_cates = [' ' + each for each in self.cate_dict['obj']]
        self.ent_label_token = self.tokenizer(
            obj_cates,
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).input_ids[:, 1:]

    def get_classifier_weight(self, label_type):
        # extract classifier weights
        all_vocab_weight = copy.deepcopy(
            self.word_embedding_weight)
        if label_type == 'rel':
            tgt_label_token = self.rel_label_token
        elif label_type == 'ent':
            tgt_label_token = self.ent_label_token

        tgt_label_token[tgt_label_token == self.tokenizer.pad_token_id] = 0
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

        tgt_label_token[tgt_label_token == self.tokenizer.pad_token_id] = 0
        pred_scores_filtered = []
        token_range = []
        for each_cate in tgt_label_token:
            pred_scores_filtered.append(
                pred_prob[:, each_cate[each_cate.nonzero().view(-1)]])
            token_range.append(len(each_cate.nonzero().view(-1)))
        all_label_score_filtered = torch.cat(pred_scores_filtered, dim=1)

        return all_label_score_filtered, token_range

    def __call__(self, pred_instances):

        ent_clser_weight, ent_cls_token_range = self.get_classifier_weight(
            'ent')
        rel_clser_weight, pred_cls_token_range = self.get_classifier_weight(
            'rel')
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

            self.word_embedding_predictor.to(all_obj_pred_hs.device)

            all_obj_pred_scores = self.word_embedding_predictor(
                all_obj_pred_hs).softmax(dim=-1)
            all_sub_pred_scores = self.word_embedding_predictor(
                all_sub_pred_hs).softmax(dim=-1)
            all_pred_pred_scores = self.word_embedding_predictor(
                all_pred_pred_hs).softmax(dim=-1)

            obj_pred_scores, _ = self.get_classifier_scores(
                all_obj_pred_scores, 'ent')
            sub_pred_scores, _ = self.get_classifier_scores(
                all_sub_pred_scores, 'ent')
            pred_pred_scores, _ = self.get_classifier_scores(
                all_pred_pred_scores, 'rel')

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

            num_cate = 30
            for inst_id in range(len(scores_all['obj'])):
                for fld_name in ['obj', 'sub', 'predicate']:
                    pred_instances[bi][inst_id][fld_name]['scores_mapped'] = scores_all[fld_name][inst_id]
                    pred_instances[bi][inst_id][fld_name]['labels_mapped'] = torch.topk(scores_all[fld_name][inst_id],
                                                                                        k=num_cate, dim=-1)[1]

        return pred_instances


class Sequence2Instance:
    def __init__(self, prompt_length, cate_dict, tokenizer, 
                 top_k_label_num, top_k_predicate_label_num,
                 border_tokens=['[obj]', '[pad]', '[sep]', '</s>', '<pad>', '\n', '<0x0A>']):
        self.prompt_length = prompt_length
        self.cate_dict = cate_dict
        self.tokenizer = tokenizer
        self.top_k_label_num = top_k_label_num
        self.top_k_predicate_label_num = top_k_predicate_label_num
        self.device = None
        self.border_tokens = border_tokens

    def __call__(self, decoder_out, raw_caption, prompt_length=None, decoder_hidden_states=None, verbose=False):
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
            seq_no_pmpt_filtered = [
                'None' if each is None else each for each in seq_no_pmpt]
            seq_no_pmpt = seq_no_pmpt_filtered

            tok_seq_nppmpt = decoder_out['sequences'][bi][prompt_length:]
            seq_len.append(len(seq_no_pmpt))
            tok_hs_dump = None
            if decoder_hidden_states is not None:
                tok_hs_dump = []
                for l_i, l in enumerate(decoder_hidden_states):
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

            if verbose:
                print(seq_no_pmpt)

            while curr_t_idx < len(seq_no_pmpt):
                tok_word = seq_no_pmpt[curr_t_idx]
                if tok_word is None:
                    tok_word = 'None'
                    seq_no_pmpt[curr_t_idx] = tok_word

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
                        seq_no_pmpt, self.border_tokens, start_p, curr_t_idx)

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
                        sub_tok_hs = torch.ones((1, 768,)).to(self.device)
                        sub_int_tok_hs = torch.ones((1, 768,)).to(self.device)

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
                            print('invalid predicate name',
                                  predicate_label_name)
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
                        predicate_tok_hs = torch.ones(
                            (1, 768,)).to(self.device)

                    # get object
                    find_end = len(seq_no_pmpt) if curr_t_idx + \
                        8 > len(seq_no_pmpt) else curr_t_idx+8
                    last_obj_b = find_token(
                        seq_no_pmpt, self.border_tokens, curr_t_idx, find_end)

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
                        obj_tok_hs = torch.ones((1, 768,)).to(self.device)
                        obj_int_tok_hs = torch.ones((1, 768,)).to(self.device)

                    if check_label_name(obj_label_name):
                        # print("corrput obj_label_name", obj_label_name)
                        bad_inst += 1
                        curr_t_idx = curr_t_idx + 1

                        if verbose:
                            print('invaild object name', obj_label_name)
                        continue

                    # if sub_label_name[0] == 'and' or obj_label_name[0] == 'and':

                    new_inst = {"sub": {"label": self.tokenizer.clean_text_from_decode(sub_label_name).strip(' '),
                                        "label_tkn": sub_label_token,
                                        "token_start": sub_ent_start,
                                        "token_end": sub_ent_end,
                                        "pred_scores": sub_pred_scores.detach().cpu(),
                                        'pred_dist': sub_pred_scores_dist,
                                        'dec_hs': sub_tok_hs,
                                        'int_tok_hs': sub_int_tok_hs, },
                                'obj': {"label": self.tokenizer.clean_text_from_decode(obj_label_name).strip(' '),
                                        "label_tkn": obj_label_token,
                                        "token_start": obj_ent_start,
                                        "token_end": obj_ent_end,
                                        "pred_scores": obj_pred_scores.detach().cpu(),
                                        'pred_dist': obj_pred_dist,
                                        'dec_hs': obj_tok_hs,
                                        'int_tok_hs': obj_int_tok_hs, },
                                "predicate": {"label": self.tokenizer.clean_text_from_decode(predicate_label_name).strip(' '),
                                              "label_tkn": predicate_label_token,
                                              "pred_scores": predicate_pred_scores,
                                              'pred_dist': predicate_pred_scores_dist,
                                              "token_start": pred_label_start,
                                              "token_end": pred_label_end,
                                              'dec_hs': predicate_tok_hs, }}
                    if verbose:
                        print(new_inst['sub']['label'], new_inst['predicate']
                              ['label'], new_inst['obj']['label'])
                        print(sub_label_name, sub_label_token, sub_pred_scores)
                        print(predicate_label_name,
                              predicate_label_token, predicate_pred_scores)
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

            if len(object_list) <= 0 :
                object_list.append(
                   {"sub": {"label": "none",
                                        "label_tkn": torch.ones((1,)),
                                        "token_start": 0,
                                        "token_end": 0,
                                        "pred_scores": torch.ones((1,)),
                                        'pred_dist': torch.ones((1,)),
                                        'dec_hs': torch.ones((1, 768,)).to(self.device),
                                        'int_tok_hs': torch.ones((1, 768,)).to(self.device), },
                                'obj': {"label": "none",
                                        "label_tkn": torch.ones((1,)),
                                        "token_start": 0,
                                        "token_end": 0,
                                        "pred_scores": torch.ones((1,)),
                                        'pred_dist': torch.ones((1,)),
                                        'dec_hs': torch.ones((1, 768,)).to(self.device),
                                        'int_tok_hs': torch.ones((1, 768,)).to(self.device), },
                                "predicate": {"label": "none",
                                              "label_tkn": torch.ones((1,)),
                                              "pred_scores": torch.ones((1,)),
                                              'pred_dist': torch.ones((1,)),
                                              "token_start": 0,
                                              "token_end": 0,
                                              'dec_hs': torch.ones((1, 768,)).to(self.device), }}
                )

            batch_object_list.append(object_list)

        logger.info(
            f"instance det stat {np.mean(vaild_inst_nun):.1f}/{np.mean(all_inst_num):.1f}\n seq_len {np.mean(seq_len):.1f}")
        return batch_object_list

    @torch.no_grad()
    def postprocess(self, batch_pred_instances, batch_targets_instance):
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
                        init_cate_id = index(cate_dict, gen_words.strip(' '))
                        gen_idx = init_gen_idx.clone()
                        matched = True
                        if gen_idx.item() != init_cate_id:
                            matched = False
                            if init_cate_id != -1:
                                gen_idx = init_cate_id
                            else:
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
                                            gen_idx=None):
                        filter_value = 1e-20

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

                        logits = pred_probs.log()
                        logits[pred_probs < filter_value * 2] = -999
                        pred_probs = logits.softmax(-1)

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

                        samp_weight_probs = pred_probs
                        ampl_scale = 4
                        if gen_idx is not None:
                            head_weight = pred_probs.sum() * ampl_scale
                            samp_weight_probs = pred_probs * 0.5
                            samp_weight_probs[gen_idx] = head_weight

                        pred_labels = torch.multinomial(
                            samp_weight_probs, num_samples=select_range)

                        # if gen_idx is not None:
                        #     samp_weight_probs[gen_idx] /= ampl_scale

                        logits = samp_weight_probs.log()
                        logits[samp_weight_probs <
                               filter_value * 2] = float('-inf')
                        pred_scores = logits.softmax(-1)

                        return pred_labels, pred_scores, samp_weight_probs

                    def sample_labels(select_range, probs):
                        pred_labels = torch.multinomial(
                            probs, num_samples=select_range, replacement=True)
                        return pred_labels, probs

                    obj_labels, obj_scores, obj_sample_prob = topk_score_selected(all_pair_num, obj_scores,
                                                                                  topk=int(len(obj_scores) * 0.6), top_k_filter=True,
                                                                                  top_p_filter=False, gen_idx=obj_gen_idx)
                    sub_labels, sub_scores, sub_sample_prob = topk_score_selected(all_pair_num, sub_scores,
                                                                                  topk=int(len(obj_scores) * 0.6), top_k_filter=True,
                                                                                  top_p_filter=False, gen_idx=sub_gen_idx)
                    predicate_labels, predicate_scores, predi_sample_prob = topk_score_selected(all_predicate_num, predicate_scores,
                                                                                                topk=int(len(predicate_scores) * 0.6), top_k_filter=True,
                                                                                                top_p_filter=False, gen_idx=predicate_gen_idx)

                    # obj_labels, obj_scores = sample_labels(all_pair_num, obj_scores)
                    # sub_labels, sub_scores = sample_labels(all_pair_num, sub_scores)
                    # predicate_labels, predicate_scores = sample_labels(all_predicate_num, predicate_scores)
                    show_match_detail = False
                    if show_match_detail:
                        print("init label and score")
                        print(each_inst['sub']['label'], each_inst['predicate']['label'], each_inst['obj']['label'], )
                        print(sub_pred_scores, predicate_pred_scores,obj_pred_scores,)

                        def top_tkn_score(pred_dist):
                            scr, token = pred_dist.sort(
                                descending=True, dim=-1)
                            words = self.tokenizer.decode(token[:, :10].T.reshape(-1))
                            scrs = scr[:, :10].reshape(-1)
                            return words, scrs
                        
                        words, scrs = top_tkn_score(sub_pred_dist)
                        print(words)
                        print(scrs)
                        print([index(self.cate_dict['obj'], each)
                              for each in words.split(' ')])

                        words, scrs = top_tkn_score(predicate_pred_dist)
                        print(words)
                        print(scrs)
                        print([index(self.cate_dict['rel'], each)
                              for each in words.split(' ')])
                        
                        words, scrs = top_tkn_score(obj_pred_dist)
                        print(words)
                        print(scrs)
                        print([index(self.cate_dict['obj'], each)
                              for each in words.split(' ')])
                        
                        init_obj_cate_id = index(
                            self.cate_dict['obj'], each_inst['obj']['label'])
                        init_sub_cate_id = index(
                            self.cate_dict['obj'], each_inst['sub']['label'])
                        init_pred_cate_id = index(
                            self.cate_dict['rel'], each_inst['predicate']['label'])
                        print(init_sub_cate_id, init_pred_cate_id, init_obj_cate_id,)

                        print()
                        print("sampled label and score:")
                        print(sub_labels, predicate_labels, obj_labels)
                        print(sub_sample_prob.sort(descending=True)[0][:all_pair_num*3], 
                              sub_sample_prob.sort(descending=True)[1][:all_pair_num*3])
                        print(predi_sample_prob.sort(descending=True)[0][:all_predicate_num*3],
                              predi_sample_prob.sort(descending=True)[1][:all_predicate_num*3])
                        print(obj_sample_prob.sort(descending=True)[0][:all_pair_num*3], 
                              obj_sample_prob.sort(descending=True)[1][:all_pair_num*3])

                        print([self.cate_dict['obj'][l_id.item()] for l_id in sub_labels], sub_labels.tolist())
                        print([self.cate_dict['rel'][l_id.item()] for l_id in predicate_labels], predicate_labels.tolist())
                        print([self.cate_dict['obj'][l_id.item()] for l_id in obj_labels], obj_labels.tolist())

                        if not sub_matched or not obj_matched or not pred_matched:
                            print(sub_gen_idx, init_sub_gen_idx)
                            print(predicate_gen_idx, init_predicate_gen_idx)
                            print(obj_gen_idx, init_obj_gen_idx)
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

                iou_threshold = 0.8
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
                    max_iou_val, match_idx = match_iou.max(-1)
                    for m_i, select_i in enumerate(all_indices):
                        new_rel_pair_idx[all_rel_tripelts ==
                                         select_i] = nms_indices[match_idx[m_i]]

                all_rel_tripelts = new_rel_pair_idx
                all_ent_label = all_ent_label_nms
                all_ent_score = all_ent_scores_nms
                all_ent_dist = all_ent_dist_nms
                all_box = all_ent_box_nms

                self_iou = generalized_box_iou(all_box, all_box)
                sub_idx = all_rel_tripelts[:, 0]
                obj_idx = all_rel_tripelts[:, 1]

                # self connection remove
                boxes_pair_iou = self_iou[sub_idx, obj_idx]
                non_self_conn_idx = torch.logical_and(boxes_pair_iou < 0.85,
                                                      all_ent_label[sub_idx] != all_ent_label[obj_idx])
                all_rel_tripelts = all_rel_tripelts[non_self_conn_idx]
                all_pred_labels_sorted = all_pred_labels_sorted[non_self_conn_idx]
                triplets_scores = triplets_scores[non_self_conn_idx]
                all_predicates_scores_sorted = all_predicates_scores_sorted[non_self_conn_idx]
                all_predicates_dist_sorted = all_predicates_dist_sorted[non_self_conn_idx]

                # triplets NMS
                pred_rel_triplet_selected, bin_mask = rel_prediction_filtering(
                    all_rel_tripelts, sub_idx, obj_idx,
                    all_ent_label, all_pred_labels_sorted, triplets_scores,
                    self_iou, overlap_thres=0.8
                )

                _, triplets_indx = all_predicates_scores_sorted[bin_mask].sort(
                    dim=0, descending=True)

                # print("rel_prediction_filtering", time.perf_counter()-start)
                # print(len(pred_rel_triplet_selected), len(bin_mask))

                prediction = BoxList(all_box,
                                     (img_orig_size[1], img_orig_size[0]), mode="xyxy")

                prediction.add_field('pred_labels', all_ent_label)
                prediction.add_field('pred_scores', all_ent_score)
                prediction.add_field('pred_dists', all_ent_dist)

                prediction.add_field(
                    'rel_pair_idxs', pred_rel_triplet_selected[triplets_indx])  # N, 2
                prediction.add_field(
                    'pred_rel_dist', all_predicates_dist_sorted[bin_mask][triplets_indx])
                prediction.add_field(
                    'pred_rel_score', all_predicates_scores_sorted[bin_mask][triplets_indx])
                prediction.add_field(
                    'pred_rel_label', all_pred_labels_sorted[bin_mask][triplets_indx])
                prediction.add_field('pred_rel_trp_score',
                                     triplets_scores[bin_mask][triplets_indx])

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
