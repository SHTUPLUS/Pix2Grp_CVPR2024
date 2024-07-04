"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import copy
import json
import torch
import torch.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.bert.tokenization_bert import BertTokenizer

from lavis.common.registry import registry
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder
from lavis.tasks.evaluation.boxlist import BoxList


SPEICAL_TOKEN_NUM = 10


SPEC = -2
WORD = -3
COORD = -4
NOTIF = -5



@registry.register_model("blip_detection")
class BlipDetection(BlipBase):
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
        "base_coco": "configs/models/blip_det_base_coco.yaml",
        "base_oiv6": "configs/models/blip_det_base_oiv6.yaml",
        "base_gqa": "configs/models/blip_det_base_gqa.yaml",
    }


    def __init__(self, image_encoder, text_decoder, reduction='mean', prompt=None, 
                 max_txt_len=40, num_coord_bin=1000, top_k_label_num=4, aux_close_classifier=False,
                 cate_dict_url=""):
        super().__init__()

        self.tokenizer, self.text_vocab_size, self.num_sepecial_tokens = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.max_txt_len = max_txt_len
        self.num_coord_bins =  num_coord_bin
        self.coord_token_start = self.text_vocab_size + SPEICAL_TOKEN_NUM + 1
        self.dump_pred = False
        self.aux_close_classifier = aux_close_classifier

        self.loss_reduction = reduction

        self.top_k_label_num = top_k_label_num

        self.cate_dict_url = cate_dict_url

        self.get_category()

    @classmethod
    def init_tokenizer(cls):
        tokenizer = StageBertTokenizer.from_pretrained("/public/home/lirj2/projects/LAVIS_GITM/data/bert-base-uncased",
                                                        local_files_only=True)
        vocab_size = len(tokenizer)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

        tokenizer.add_tokens(["[CORD]"])
        tokenizer.obj_coord_token_id = len(tokenizer) - 1
        tokenizer.add_tokens(["[OBJ]"])
        tokenizer.obj_s_token_id = len(tokenizer) - 1
        tokenizer.add_tokens([ "[OBJ/]"])
        tokenizer.obj_e_token_id = len(tokenizer) - 1
        tokenizer.add_tokens([ "[NOS]"])
        tokenizer.noise_token_id = len(tokenizer) - 1

        special_token_num = len(tokenizer) - vocab_size
        print("special_token_num", special_token_num)

        return tokenizer, vocab_size, special_token_num

    def get_category(self):
        with open(self.cate_dict_url, 'r') as f:
            self.cate_dict = json.load(f)

        self.ent_label_token = self.tokenizer(
            self.cate_dict['obj'],
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).input_ids[:, 1:]

        if self.aux_close_classifier:
            self.close_classifier = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.ReLU(),
                    nn.Linear(768, len(self.cate_dict['obj'])),
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

        for bi, b_pred_inst in enumerate(pred_instances):
            obj_ent_hs = []
            obj_pred_token_range = []
            for each_inst in b_pred_inst:
                obj_ent_hs.append(each_inst['dec_hs'])
                obj_pred_token_range.append(len(each_inst['dec_hs']))
            
            if len(obj_ent_hs) == 0:
                continue

            all_obj_pred_hs = torch.cat(obj_ent_hs, dim=0)
            all_obj_pred_scores = self.text_decoder.cls.predictions(all_obj_pred_hs).softmax(dim=-1)
            obj_pred_scores, _ = self.get_classifier_scores(all_obj_pred_scores, 'ent')

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

            scores_all = range_sum(obj_pred_scores, ent_cls_token_range)
            token_range_all = obj_pred_token_range

            scores_all = range_sum(
                    scores_all.T, token_range_all).T

            for inst_id in range(len(scores_all)):
                    pred_instances[bi][inst_id]['scores_mapped'] = scores_all[inst_id]

                    # pred_instances[bi][inst_id][fld_name]['labels_mapped'] = torch.multinomial(scores_all[fld_name][inst_id], num_samples=5)
                    # print(pred_instances[bi][inst_id][fld_name]['labels_mapped'])
                    
                    # pred_instances[bi][inst_id][fld_name]['labels_mapped'] = torch.multinomial(
                    #     torch.softmax(scores_all[fld_name][inst_id], dim=-1), num_samples=5)
                    # print(pred_instances[bi][inst_id][fld_name]['labels_mapped'])

                    pred_instances[bi][inst_id]['labels_mapped'] = torch.topk(scores_all[inst_id], 
                                                                                        k=self.top_k_label_num, dim=-1)[1]
                    # print(pred_instances[bi][inst_id][fld_name]['labels_mapped'])
                    # print()

        return pred_instances

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

        all_pair_num = 2
        all_predicate_num = 2

        for bi in range(len(batch_pred_instances)):
            # [h, w] LAVIS/lavis/datasets/datasets/oiv6_rel_detection.py:216
            img_orig_size = batch_targets_instance[bi]['orig_size'].cpu(
            ).tolist()

            image_info.append({
                "img_orig_size": img_orig_size,
                "image_id": batch_targets_instance[bi]['image_id']
            })

            inst_pred = batch_pred_instances[bi]

            def xywh2xyxy(boxes: torch.Tensor):
                boxes_xyxy = copy.deepcopy(boxes)
                boxes_xyxy[:, :2], boxes_xyxy[:, 2:] = boxes[:, :2] - \
                    boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2
                return boxes_xyxy.int()
            
            def xywh2xyxy(boxes: torch.Tensor):
                boxes_xyxy = copy.deepcopy(boxes)
                boxes_xyxy[:, :2], boxes_xyxy[:, 2:] = boxes[:, :2] - \
                    boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2
                boxes_xyxy[:, 0::2] *= img_orig_size[1]
                boxes_xyxy[:, 1::2] *= img_orig_size[0]
                return boxes_xyxy.int()

            def rel2abs(boxes: torch.Tensor):
                boxes_xyxy = copy.deepcopy(boxes)
                boxes_xyxy[:, 0::2] *= img_orig_size[1]
                boxes_xyxy[:, 1::2] *= img_orig_size[0]
                return boxes_xyxy.int()
        

            if len(inst_pred) > 0:
                # ranking all relationship
                all_obj_labels = []
                all_obj_scores = []

                all_obj_dist = []

                all_obj_box = []

                init_inst_idx = []
                for idx, each_inst in enumerate(inst_pred):
                    obj_score_dist = each_inst['scores_mapped']

                    obj_labels = each_inst['labels_mapped']
                    # todo may insuffcient repeating all_pair_num * all_predicate_num * all_predicate_num
                    obj_scores = torch.index_select(obj_score_dist, -1, obj_labels)

                    all_obj_scores.append(obj_scores)

                    all_obj_dist.append(obj_score_dist)
                    all_obj_box.append(each_inst['boxes'])
                    all_obj_labels.append(obj_labels)
                    init_inst_idx.append(torch.ones(obj_labels.shape[0], dtype=int)*idx)

                all_obj_scores =  torch.cat(all_obj_scores)
                all_obj_scores_sorted, object_indx = all_obj_scores.sort(
                    dim=0, descending=True)

                all_obj_labels_sorted = torch.cat(all_obj_labels)[object_indx]

                # unexpand fields
                init_inst_idx_sorted = torch.cat(init_inst_idx)[object_indx]
                all_obj_dist_sorted = torch.stack(all_obj_dist)[init_inst_idx_sorted]
                all_obj_box_sorted = torch.stack(all_obj_box)[init_inst_idx_sorted]

                # fill data format
                prediction = BoxList(rel2abs(all_obj_box_sorted),
                                    (img_orig_size[1], img_orig_size[0]), mode="xyxy")

                prediction.add_field('pred_labels',all_obj_labels_sorted)
                prediction.add_field('pred_scores', all_obj_scores_sorted)
                prediction.add_field('pred_dists', all_obj_dist_sorted)
                prediction.to(torch.device('cpu'))
                predictions.append(prediction)
            else:
                #  padding instance
                prediction = BoxList(torch.zeros((1,4)), 
                                    (img_orig_size[1], img_orig_size[0]), mode="xyxy")

                prediction.add_field('pred_labels', torch.zeros((1)).int())
                prediction.add_field('pred_scores', torch.zeros((1)))
                prediction.add_field('pred_dists', torch.zeros((1, len(self.cate_dict['obj']))))
                prediction.to(torch.device('cpu'))

                predictions.append(prediction)

            gt_boxes = batch_targets_instance[bi]['boxes']
            groundtruth = BoxList( xywh2xyxy(gt_boxes), (img_orig_size[1], img_orig_size[0]), mode="xyxy")
            groundtruth.add_field('labels', batch_targets_instance[bi]['det_labels'])
            groundtruth.to(torch.device('cpu'))
            ground_truths.append(groundtruth)

        return predictions, ground_truths, image_info


    def forward_encoder(self, samples):
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        return image_embeds


    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder
        targets = samples["targets"]

        label_token, decoder_targets = self.target2token_seqs(targets)

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        decoder_output = self.text_decoder(
            input_ids=label_token.input_ids,
            attention_mask=label_token.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            reduction=self.loss_reduction,
            labels=decoder_targets,
            return_dict=True,
        )

        if self.aux_close_classifier:
            batch_object_list = self.seq2instances_train(decoder_output, targets, decoder_targets)
            decoder_output = self.vocab2category_train(batch_object_list, decoder_output, decoder_targets)

        return decoder_output, decoder_targets

    def seq2instances_train(self, decoder_output, targets, decoder_targets):

        pass
    
    def vocab2category_train(self, batch_object_list, decoder_output, decoder_targets):
        pass

    def target2token_seqs(self, targets):
        tar_seqs = []
        for b_i, target in enumerate(targets):

            all_idx = torch.randperm(len(target['boxes'])).tolist()
            target['boxes'] = target['boxes'][all_idx]
            target['labels_texts'] = [target['labels_texts'][i] for i in all_idx]
        
            labels_text = target['labels_texts']
            tar_seq = self.prompt + "[OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/]".join(labels_text) + "[OBJ] [CORD] [CORD] [CORD] [CORD] [OBJ/]" 
            tar_seqs.append(tar_seq)

        label_token = self.tokenizer(
            tar_seqs,
            # padding="max_length",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        # all_cates = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        # all_cates_token = self.tokenizer(
        #     " ".join(all_cates),
        #     # padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(self.device)

        label_token.input_ids[label_token.input_ids == 102] = 0
        label_token.input_ids[:, 0] = self.tokenizer.bos_token_id

        raw_token = []

        for b_i, target in enumerate(targets):
            
            boxes = target['boxes']
            boxes_xyxy = boxes[:]
            boxes_xyxy[:,:2], boxes_xyxy[:,2:] = boxes[:,:2] - boxes[:,2:]/2, boxes[:,:2] + boxes[:,2:]/2 ## (x1,y1,x2,y2)

            box_tokens = (boxes * self.num_coord_bins).floor().long().clamp(min=0, max=self.num_coord_bins) + self.coord_token_start

            object_idx = 0
            coord_id = 0
            label_token_fuse = label_token['input_ids'][b_i, :]
            for tok_idx, each in enumerate(label_token_fuse):
                if each == self.tokenizer.obj_e_token_id or coord_id >= 4:
                    object_idx += 1
                    coord_id = 0

                if each == self.tokenizer.pad_token_id or object_idx >= len(box_tokens):  
                    break

                if each == self.tokenizer.obj_coord_token_id:
                    label_token_fuse[tok_idx] = box_tokens[object_idx][coord_id]
                    coord_id += 1

            raw_token.append(label_token_fuse)
        raw_token = torch.stack(raw_token)
        label_token.input_ids = raw_token

        # prepare targets for forwarding decoder
        decoder_targets = label_token.input_ids.masked_fill(
            label_token.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        return label_token,decoder_targets

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
        decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)

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
        use_nucleus_sampling=False,
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
        # odict_keys(['sequences', 'decoder_attentions', 'cross_attentions'])
        # decoder_out.cross_attentions: 
        #       seq_len, decoder_layer, batch_size*num_captions, head_num, previous_seq_size(prompt_size or 1), image_hs_size

        raw_caption = [self.tokenizer.decode(decoder_out.sequences[i].detach().cpu(), decode_raw_tokens=True, skip_special_tokens=True)
                          for i in range(len(decoder_out.sequences))]

        # from token to instance
        batch_object_list = self.seq2instance(decoder_out, raw_caption)
        # mapping prediction into fix label space
        batch_object_list_cate_trans = self.vocab2category(batch_object_list)

        predictions, ground_truths, image_info = self._postprocess(
            batch_object_list_cate_trans, samples['targets'])

        if self.dump_pred:
            image_ids = samples['instance_id'][0]
            
            forward_buffer = {
                'cross_attentions': decoder_out['cross_attentions'],
                'decoder_attentions': decoder_out['decoder_attentions'],
                'scores': decoder_out['scores'], # num_toke, bz, vocab_size
                'raw_token': decoder_out.sequences.detach().cpu(),
                'raw_caption': raw_caption,
                'image_path': samples['image_pth'],
                'normed_image': samples['image'], 
                "batch_object_list": batch_object_list,
                "gt_instance": samples['targets']
                # 'image_pixel': data_loader.dataset.__getitem__(img_idx, raw_image=True),
                # 'image_size': (image_size, patch_size)
            }
            import pickle
            with open(f'/mnt/petrelfs/lirongjie/project/LAVIS/lavis/output/BLIP/detection_coco/vis_dump/vl_det_dump_{image_ids}.pkl', 'wb') as f:
                print(f"save data {image_ids}")
                pickle.dump(forward_buffer, f)
            if image_ids > 48:
                exit()

        return predictions, ground_truths, image_info

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoderDethead.from_config(cfg)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 500)

        model = cls(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len,
                     num_coord_bin=cfg.get("num_coord_bin", 1000),
                     reduction=cfg.get("reduction", 'mean'),
                     top_k_label_num=cfg.get("top_k_label_num", 5),
                     cate_dict_url=cfg.get("cate_dict_url", "cache/openimages/open-imagev6/annotations/categories_dict.json"),
                     )
        model.load_checkpoint_from_config(cfg)

        return model
    
    def seq2instance(self, decoder_out, raw_caption):
        batch_object_list = []
        for bi, seq in enumerate(raw_caption):
            object_list = []
            last_start = 0
            seq_no_pmpt = seq[4:]
            tok_seq_nppmpt = decoder_out.sequences[bi][4:]
            scores_tensor = []
            for each in decoder_out['scores']:
                scores_tensor.append(each[bi].softmax(dim=-1))

            tok_hs_dump = None
            if decoder_out.get('decoder_hidden_states') is not None:
                tok_hs_dump = []
                for l_i, l in enumerate(decoder_out['decoder_hidden_states']):
                    tok_hs_dump.append(l[-1][bi][-1])
                tok_hs_dump = torch.stack(tok_hs_dump)


            pred_score = torch.stack(scores_tensor) # num_tok, vocab_size

            last_start = 0
            all_inst = 0
            bad_inst = 0
            curr_t_idx = 0
            

            while curr_t_idx < len(seq_no_pmpt):
                tok = seq_no_pmpt[curr_t_idx]
                if tok == '[obj]':
                    all_inst += 1

                    last_obj_e = find_token(
                        seq_no_pmpt, '[obj/]', curr_t_idx, curr_t_idx + 6)
                    last_obj_b = find_token(
                        seq_no_pmpt, '[obj]', last_start, curr_t_idx)

                    if len(last_obj_e) < 1 :
                        # print('subject cropput')
                        bad_inst += 1
                        # update index
                        curr_t_idx = curr_t_idx + 1
                        continue

                    if len(last_obj_b) >= 1:
                        last_start = last_obj_b[-1]

                    box_coord_b = curr_t_idx
                    box_coord_e = last_obj_e[0]

                    if box_coord_b > box_coord_e or box_coord_e - box_coord_b != 5:
                        obj_boxs = tok_seq_nppmpt[box_coord_b +1: box_coord_e].detach().cpu()
                        # obj_boxs = (obj_boxs - self.coord_token_start) / self.num_coord_bins
                        # last_start = curr_t_idx + 1
                        # print('subject cropput', obj_boxs)
                        if len(obj_boxs) > 4:
                            # print('fixed as', obj_boxs[:4])
                            box_coord_e = box_coord_b + 5
                        else:
                            bad_inst += 1
                            last_start = box_coord_e + 1
                            curr_t_idx = last_start + 1
                            continue

                    if len(last_obj_e) >= 2:
                        last_start = last_obj_e[-2] + 1

                    ent_cate_pred_start = last_start
                    if seq_no_pmpt[ent_cate_pred_start] == 'and':
                        ent_cate_pred_start += 1

                    ent_cate_pred_end = box_coord_b
                    ent_box_start = box_coord_b + 1
                    ent_box_end = box_coord_e

                    sub_ent_start = ent_cate_pred_start

                    sub_label_name = seq_no_pmpt[ent_cate_pred_start: ent_cate_pred_end]
                    sub_label_token = tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]
                    if pred_score is not None:
                        sub_pred_scores = pred_score[ent_cate_pred_start: ent_cate_pred_end,
                                                     tok_seq_nppmpt[ent_cate_pred_start: ent_cate_pred_end]]
                    else:
                        sub_pred_scores = torch.ones((1,))

                    sub_tok_hs = None
                    if tok_hs_dump is not None:
                        sub_tok_hs = tok_hs_dump[ent_cate_pred_start: ent_cate_pred_end]
                    else:
                        sub_tok_hs = torch.ones((768,))

                    sub_boxs = tok_seq_nppmpt[ent_box_start: ent_box_end].detach(
                    ).cpu()
                    sub_boxs = (sub_boxs - self.coord_token_start) / \
                        self.num_coord_bins

                    if check_label_name(sub_label_name):
                        # print("corrput sub_label_name", sub_label_name)
                        bad_inst += 1
                        # update index
                        last_start = ent_cate_pred_end + 1
                        curr_t_idx = last_start + 1
                        continue

                    object_list.append({"label":" ".join(sub_label_name), 
                                        "label_tkn": sub_label_token,
                                        "boxes": sub_boxs, 'dec_hs': sub_tok_hs,
                                        "token_start": sub_ent_start, 
                                        "pred_scores":sub_pred_scores.detach().cpu()})
                    last_start = last_obj_e[0] + 1
                    curr_t_idx = curr_t_idx + 1
                
                else:
                    curr_t_idx += 1

                if bad_inst > 40:
                    break

            # print(f"all instance det {all_inst}, valid instance det {all_inst - bad_inst}")
            batch_object_list.append(object_list)
        return batch_object_list



class StageBertTokenizer(BertTokenizer):

    def _decode(
        self,
        token_ids: list,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        if kwargs.pop("decode_raw_tokens", False):
            return self.convert_ids_to_tokens(token_ids)

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
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
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
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
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


def find_token(seq, target, s, e):
    target_indx = []
    if len(seq) < e:
        e = len(seq)
    if s < 0:
        s = 0
        
    if isinstance(target, str):
        for i in range(s, e):
            if seq[i] is not None:
                if seq[i].upper() == target.upper():
                    target_indx.append(i)
        return target_indx
    elif isinstance(target, list):
        for i in range(s, e):
            for each_t in target:
                if seq[i] is not None:
                    if seq[i].upper() == each_t.upper():
                        target_indx.append(i)
                        break
        return target_indx

def check_label_name(label_name):
    if len(label_name) <= 0:
        return True
    invalid_det_label_pred = False
    for each in label_name:
        if each.upper() in ['[obj/]'.upper(),  '[SEP]', '[UNK]', '[obj]'.upper(), '[REL]']:
            invalid_det_label_pred = True
    return invalid_det_label_pred

class XBertLMHeadDecoderDethead(XBertLMHeadDecoder):
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

        outputs = self.bert( # BertModel
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

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()
        
        lm_loss = None
        loss_dict = {}

        def get_token_type(labels):
            token_types = torch.zeros_like(labels)
            token_types[torch.logical_and(labels >= 103, labels < 30524)] = WORD
            token_types[torch.logical_and(labels >= 30524, labels < 30534)] = NOTIF
            token_types[labels >= 30534] = COORD
            return token_types
        
        # we are doing next-token prediction; shift prediction scores and input ids by one
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            token_type = get_token_type(labels)
            num_pos = (token_type == WORD).sum(-1)

            label_smoothing = 0.01
            if reduction == "mean":
                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction=reduction, label_smoothing=label_smoothing)
                lm_loss = loss_fct(
                    shifted_prediction_scores.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )

                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none", label_smoothing=label_smoothing, ignore_index=-100)
                loss_none = loss_fct(shifted_prediction_scores.permute(
                    0, 2, 1), labels)  # batch_size, seq
                weighted_lm_loss = loss_none
                word_loss = (
                    weighted_lm_loss[token_type == WORD].sum() / num_pos).mean()
                coord_loss = (
                    weighted_lm_loss[token_type == COORD].sum() / num_pos).mean()
                notif_loss = (
                    weighted_lm_loss[token_type == NOTIF].sum() / num_pos).mean()

                word_acc = sum(shifted_prediction_scores[token_type == WORD].max(-1)[
                               1] == labels[token_type == WORD]) / weighted_lm_loss[token_type == WORD].view(-1).shape[0]
                notif_acc = sum(shifted_prediction_scores[token_type == NOTIF].max(-1)[
                                1] == labels[token_type == NOTIF]) / weighted_lm_loss[token_type == NOTIF].view(-1).shape[0]
                coord_error = sum(torch.abs(shifted_prediction_scores[token_type == COORD].max(-1)[
                                  1] - labels[token_type == COORD])) / (weighted_lm_loss[token_type == COORD].view(-1).shape[0] * 4 )

            if reduction == "none":
                num_pos = (token_type == WORD).sum(-1)
                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none", label_smoothing=0.1, ignore_index=-100)
                loss_none = loss_fct(shifted_prediction_scores.permute(
                    0, 2, 1), labels)  # batch_size, seq

                weights = torch.ones(labels.shape).to(loss_none.device)
                weights[labels == -100] = 0.00  # padding
                weighted_lm_loss = loss_none * weights
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
                                  1] - labels[token_type == COORD])) / (weighted_lm_loss[token_type == COORD].view(-1).shape[0] * 4)

            word_loss = word_loss.mean()
            coord_loss = coord_loss.mean()
            notif_loss = notif_loss.mean()

            loss_dict = {
                'loss': lm_loss,
                'word_loss': word_loss,
                "coord_loss": coord_loss,
                "notif_loss": notif_loss,
                "notif_acc": notif_acc,
                "word_acc": word_acc,
                'coord_error': coord_error,
            }

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

        image_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        model_kwargs = {
            "encoder_hidden_states": visual_embeds,
            "encoder_attention_mask": image_atts,
        }
        model_kwargs.update(kwargs)

        outputs = self.generate(
            input_ids=tokenized_prompt.input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            eos_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=1.5,
            **model_kwargs
        )

        return outputs