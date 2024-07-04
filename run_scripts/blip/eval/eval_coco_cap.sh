python -m torch.distributed.run --master_port 13455 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/caption_coco_eval.yaml
