python -m torch.distributed.run --master_port 13958 --nproc_per_node 1 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_gqa_eval.yaml --job-name $1 
