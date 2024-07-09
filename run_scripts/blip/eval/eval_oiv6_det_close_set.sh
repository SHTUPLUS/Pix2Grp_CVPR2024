python -m torch.distributed.run --master_port 13497 --nproc_per_node 4 evaluate.py --cfg-path lavis/projects/blip/eval/det_oiv6_eval_close_set.yaml
