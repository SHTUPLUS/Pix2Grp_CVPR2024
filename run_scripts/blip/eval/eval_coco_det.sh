python -m torch.distributed.run --master_port 13455 --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip/eval/det_coco_eval.yaml
