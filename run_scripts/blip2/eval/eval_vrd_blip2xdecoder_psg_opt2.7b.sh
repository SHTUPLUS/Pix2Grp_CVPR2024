python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip2/eval/vrd_psg_opt2.7b_eval.yaml --job-name $1
