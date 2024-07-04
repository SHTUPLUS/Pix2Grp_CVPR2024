python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/vrd_psg_ft_xdecoder.yaml --job-name $1

# --git-commit