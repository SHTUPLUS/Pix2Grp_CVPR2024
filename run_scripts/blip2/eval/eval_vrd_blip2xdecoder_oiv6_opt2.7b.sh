python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2/eval/vrd_oiv6_opt2.7b_eval.yaml --job-name $1
