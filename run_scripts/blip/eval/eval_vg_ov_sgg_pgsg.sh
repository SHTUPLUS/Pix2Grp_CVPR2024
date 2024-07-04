pkill -f lets; pkill -f python
python -m torch.distributed.run --master_port 13998 --nproc_per_node $2 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_vg_pgsg_eval_ov.yaml --job-name $1 
