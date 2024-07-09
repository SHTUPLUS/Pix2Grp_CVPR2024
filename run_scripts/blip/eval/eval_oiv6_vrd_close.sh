python -m torch.distributed.run --master_port 13958 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_oiv6_eval.yaml --job-name "close_class-$1"
