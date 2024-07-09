pkill -f lets; pkill -f python

python -m torch.distributed.run --master_port 13958 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_oiv6_pgsg_ov_eval.yaml --job-name $1 

occ_script