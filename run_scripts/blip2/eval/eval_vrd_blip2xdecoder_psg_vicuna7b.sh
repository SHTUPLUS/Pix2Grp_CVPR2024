python -m torch.distributed.run --nproc_per_node=1 --master_port 13619 evaluate.py --cfg-path lavis/projects/blip2/eval/vrd_psg_vicuna7b_eval.yaml --job-name $1
