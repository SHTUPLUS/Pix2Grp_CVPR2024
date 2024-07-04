python -m torch.distributed.run --master_port 13766 --nproc_per_node=8 train.py  --cfg-path lavis/projects/blip/train/vrd_gqa_ft_xdecoder.yaml --job-name $1
