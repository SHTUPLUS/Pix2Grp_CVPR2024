python -m torch.distributed.run --master_port 13958 --nproc_per_node 8 evaluate.py --cfg-path lavis/projects/blip/eval/rel_det_gqa_xdecoder_eval.yaml --job-name $1 
