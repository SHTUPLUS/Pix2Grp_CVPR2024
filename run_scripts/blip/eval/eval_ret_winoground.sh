python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip/eval/ret_winoground_eval.yaml
# python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/blip/eval/ret_coco_eval.yaml
