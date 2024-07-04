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

from transformers import LlamaTokenizer
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from lavis.models.blip2_models.blip2_opt_vrd_xdecoder import DynamicClassifier, PosAdapter, Sequence2Instance

from lavis.models.blip_models.blip_det import COORD, NOTIF, WORD, check_label_name, find_token
from lavis.models.blip_models.blip_rel_det_pgsg import index, rel_prediction_filtering
from lavis.models.detr_transformer import MLP, PositionEmbeddingSine, build_decoder, build_encoder

from lavis.models.resnet import LightweightConv, Res18Wrapper
from lavis.models.weight_init import show_params_status
from lavis.tasks.evaluation.boxlist import BoxList


logger = logging.getLogger(__name__)


def get_token_type(labels):
    token_types = torch.zeros_like(labels)
    token_types[labels < 32000] = WORD
    token_types[labels < 0] = 0
    token_types[labels >= 32000] = NOTIF
    return token_types


@registry.register_model("blip2_vicuna_vrd")
class Blip2VicunaVrd(Blip2Base):
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
        "base_psg": "configs/models/blip2/blip2_psg_vrd_vicuna7b.yaml",

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
        llm_model="",
        prompt="",
        qformer_text_input=True,
        max_txt_len=32,
        max_output_txt_len=256,
        max_objects=99,
        max_pos_objects=20, 
        num_coord_bin=1000,
        dump_pred=False, reduction='none', 
        top_k_label_num=5, top_k_predicate_label_num=3,
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

        self.Qformer_tokenizer = self.init_tokenizer(truncation_side="left")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.qformer_text_input = qformer_text_input
        if not qformer_text_input:
            self.Qformer.cls = None

            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.Qformer_tokenizer))
        self.Qformer.cls.predictions.transform = None
        self.Qformer.cls.predictions.decoder.bias = None
        self.Qformer.cls.predictions.bias = None

        # initialize tokenizer
        self.tokenizer = StagedLlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")

        # self.tokenizer = StagedLlamaTokenizer.from_pretrained("", use_fast=False, truncation_side="left")

        self.text_vocab_size = len(self.tokenizer)
        
        self.tokenizer.add_tokens(['[OBJ]', '[REL]'])

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.tokenizer.add_special_tokens({'unk_token': '</s>'})
        
        self.ent_token_id = self.tokenizer(
            "[OBJ]", add_special_tokens=False).input_ids[0]
        self.rel_token_id = self.tokenizer(
            "[REL]", add_special_tokens=False).input_ids[0]
        self.eos_token_id = self.tokenizer(
            "</s>", add_special_tokens=False
        ).input_ids[0]
        

        # print(self.tokenizer("[OBJ]", add_special_tokens=False).input_ids[0])
        # print(self.tokenizer("[REL]", add_special_tokens=False).input_ids[0])
        
        self.llm_model = LlamaForCausalLMDetHead.from_pretrained(
            llm_model,
            # torch_dtype=torch.float16,
            torch_dtype=torch.float32,
            pos_adapter=pos_adapter, pos_adapter_conv=pos_adapter_conv,
            pos_adapter_tfmer_layer=pos_adapter_tfmer_layer,
            vis_in_dim=self.visual_encoder.num_features,
            local_files_only=True,
        )
        

        self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # init param status for VRD
        for name, param in self.llm_model.named_parameters():
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
            self.llm_model.model = get_peft_model(self.llm_model.model, lora_config)

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
                        
                for layer_id in range(len(self.llm_model.base_model.layers)-finetune_layer_num,
                                    len(self.llm_model.base_model.layers)):
                    self.llm_model.base_model.layers[layer_id] = get_peft_model(self.llm_model.base_model.layers[layer_id], lora_config)
            else:
                for layer_id in range(len(self.llm_model.base_model.layers)-finetune_layer_num,
                                    len(self.llm_model.base_model.layers)):
                    for name, param in self.llm_model.base_model.layers[layer_id].named_parameters():
                        param.requires_grad = True

        for name, param in self.llm_model.named_parameters():
            if 'pos_adapter' in name:
                param.requires_grad = True
            elif 'lm_head' in name:
                param.requires_grad = True
            elif 'embed_tokens' in name:
                param.requires_grad = True

        self.llm_model.lm_head.weight.requires_grad = True
        self.llm_model.model.embed_tokens.weight.requires_grad = True

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        # VRD adaptor modules
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
            word_embedding_predictor=self.llm_model.lm_head,
            word_embedding_predictor_weight=self.llm_model.lm_head.weight
        )
        self.dump_pred = dump_pred
        self.dump_dir = dump_dir

        self.seq2instance = Sequence2Instance(
            self.prompt_length, self.cate_dict, self.tokenizer, top_k_label_num, top_k_predicate_label_num
        )

        logger.info(show_params_status(self))

    def forward(self, samples):
        self.tokenizer.padding_side = "right"

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        prompt, query_tokens, inputs_llm, atts_llm = self.qformer_forward(samples, image_embeds, image_atts)

        outputs = self.forward_decoder(
            samples, image_embeds, inputs_llm, atts_llm, image_atts)
        loss = outputs.loss

        return {"loss": loss}

    def qformer_forward(self, samples, image_embeds, image_atts):
        bs = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt
        prompt_in = [prompt] * bs

        if self.qformer_text_input:
            text_Qformer = self.Qformer_tokenizer(
                prompt_in,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
        
        if self.qformer_text_input:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image_embeds.device)

        return prompt, query_tokens, inputs_llm, atts_llm

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

        prompt, query_tokens, inputs_llm, atts_llm = self.qformer_forward(samples, image_embeds, image_atts)

        self.tokenizer.padding_side = "left"
        prompt = [prompt] * len(atts_llm) # repeat by batch size
        llm_tokens = self.tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
        
        if use_nucleus_sampling:
            num_beams = 1
        
        decoder_out_collect = None
        seg_len = 32
        min_seq_len = 16
        
        print(max_length // seg_len + 1)

        for i in range(max_length // seg_len + 1):
            import gpustat
            print(i)
            gpustat.print_gpustat()

            decoder_out = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=seg_len,
                min_length=min_seq_len,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=True,
                output_scores=True,
            )
            # odict_keys(['sequences', 'scores', 'attentions', 'hidden_states'])
            # scores: gen_seq_len, batch_size, vocab_size
            # sequences: batch_size, seq_len
            # hidden_states: seq_len, lyr_num, batch_size, gen_uie, hidden_state
            trim_hs = []
            for gen_step, new_hs_each in enumerate(decoder_out.hidden_states):
                trim_hs.append(new_hs_each[-1][:, -1:][None, :])

            # print(len(decoder_out.hidden_states),
            #       len(decoder_out.hidden_states[4]),
            #       len(decoder_out.hidden_states[4][0]),
            #       len(decoder_out.hidden_states[4][0][0]),
            #       len(decoder_out.hidden_states[4][0][0][0]))
            # print(len(trim_hs),
            #       len(trim_hs[3]),
            #       len(trim_hs[3][0]),
            #       len(trim_hs[3][0][0]),
            #       len(trim_hs[3][0][0][0]))

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

            # outputs = self.forward_decoder(
            #     samples, image_embeds, query_output, image_atts)
            # odict_keys(['sequences', 'sequences_scores', 'scores', 'beam_indices', 'attentions', 'hidden_states'])
            # "hidden_states" :
            #       genertaion_step x decoder layer x batch_size * beam_size x (start_size + generate_step) x hid_dim

        # (forward_decoder_output,
        #     decoder_targets,
        #     box_targets) = self.forward_decoder(samples, image_embeds, inputs_llm, atts_llm, image_atts, return_box_gts_batch=True)

        decoder_out = decoder_out_collect
        self.prompt_length = llm_tokens.input_ids.shape[1]

        raw_caption = [self.tokenizer.decode(seq, skip_special_tokens=False, decode_raw_tokens=True)
                       for seq in decoder_out.sequences]
        # import ipdb; ipdb.set_trace()

        batch_object_list = self.seq2instance(decoder_out_collect, raw_caption,
                                              prompt_length=prompt_length,
                                              decoder_hidden_states=decoder_out_collect.hidden_states,
                                              verbose=False)

        batch_object_list_cate_trans = self.vocab2category(
            batch_object_list)

        if self.llm_model.pos_adapter_on:
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

            pos_ada_output = self.llm_model.pos_adapter(
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
             box_targets) = self.forward_decoder(samples, image_embeds, inputs_llm, atts_llm, image_atts, return_box_gts_batch=True)

            raw_caption_target = [self.tokenizer.decode(decoder_targets[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                  for i in range(len(decoder_targets))]
            f_dec_tokens = torch.argmax(
                forward_decoder_output.logits.contiguous(), dim=-1)
            raw_caption_fdec = [self.tokenizer.decode(f_dec_tokens[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                                for i in range(len(f_dec_tokens))]

            pos_ada_output = None
            if self.llm_model.pos_adapter_on:
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

                pos_ada_output = self.llm_model.pos_adapter(
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

    def forward_decoder(self, samples, image_embeds, inputs_opt, opt_atts, image_atts, return_box_gts_batch=False):

        # target_tokens
        if samples.get('targets') is not None:
            input_token, targets_tokens, close_vocab_classes, box_gts_batch = self.target2token_seqs(
                samples["targets"], text_mask_ratio=0.0)

            if self.prompt:
                targets_tokens[:, : self.prompt_length] = -100
                # do not apply loss to the prompt

        # input tokens
        inputs_embeds = self.llm_model.model.embed_tokens(input_token.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([opt_atts, input_token.attention_mask], dim=1)

        # print("inputs_embeds", torch.isnan(inputs_embeds).nonzero())
        # print("image_embeds", torch.isnan(image_embeds).nonzero())
        with self.maybe_autocast(): 
            decoder_output = self.llm_model(
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
            tar_seq += '.</s>'
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
        llm_model = cfg.get("llm_model")

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
            llm_model=llm_model,
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
            finetune_layer_num=cfg.get('finetune_layer_num', 4),
            qformer_text_input = cfg.get("qformer_text_input", True)
        )

        model.load_checkpoint_from_config(cfg)
        # [50264, 50265, 50266]
        # ['<mask>', '[OBJ]', '[REL]']
        # if not cfg.load_finetuned:
        #     model.llm_model.resize_token_embeddings(len(model.tokenizer))

        return model


class LlamaForCausalLMDetHead(LlamaForCausalLM):

    def __init__(self, config, pos_adapter=False, vis_in_dim=768,
                 pos_adapter_conv='none', pos_adapter_tfmer_layer=0):
        super(LlamaForCausalLMDetHead, self).__init__(config)
        self.pos_adapter_on = pos_adapter
        if self.pos_adapter_on:
            self.pos_adapter = PosAdapter(pos_adapter_conv, pos_adapter_tfmer_layer,
                                          lang_in_dim=config.hidden_size,
                                          vis_in_dim=vis_in_dim)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        prediction_scores = self.lm_head(hidden_states)

        # print("outputs[0]", torch.isnan(outputs[0]).nonzero())
        # print("prediction_scores", torch.isnan(prediction_scores).nonzero())

        lm_loss = None
        loss_dict = {}
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, -labels.size(1):-1, :].contiguous()
            # we are doing next-token prediction; shift prediction scores and input ids by one
            labels = labels[:, 1:].contiguous()
            token_type = get_token_type(labels)

            # build box loss weights
            box_l1 = []
            box_giou = []
            pos_ada_output = None
            if box_targets is not None:

                if self.pos_adapter_on:
                    sequence_output_trim = hidden_states[:, -labels.size(1) - 1:]
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
                            
                        try:
                            pred_ent_hs = torch.stack(pred_ent_hs)
                        except:
                            import ipdb; ipdb.set_trace()

                        pred_ent_hs_all.append(pred_ent_hs)

                    pos_ada_output = self.pos_adapter(
                        pred_ent_hs_all, raw_image, encoder_hidden_states, encoder_attention_mask, box_targets)

            label_smoothing = 0.001

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


        return CausalLMOutputWithPast(
            loss=loss_dict,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # cross_attentions=outputs.cross_attentions,
        )

class StagedLlamaTokenizer(LlamaTokenizer):

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

