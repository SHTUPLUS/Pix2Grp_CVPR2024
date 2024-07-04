python -m torch.distributed.run --nproc_per_node=2 --master_port 13969  evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_instruct_vicuna7b_eval.yaml
