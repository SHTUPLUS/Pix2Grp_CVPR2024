python -m torch.distributed.run --master_port 13948 --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_gqa_eval_cls_voc.yaml --job-name "close_embed_class-$1"
