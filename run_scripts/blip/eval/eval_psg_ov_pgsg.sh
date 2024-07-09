
pkill -f lets; pkill -f python

# python -m torch.distributed.run --master_port 13958 --nproc_per_node $2 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_psg_ov.yaml --job-name $1 


python -m torch.distributed.run --master_port 13958 --nproc_per_node $2 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_psg_pgsg_ov.yaml --job-name $1 