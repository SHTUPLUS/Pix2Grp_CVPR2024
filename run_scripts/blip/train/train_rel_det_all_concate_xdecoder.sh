python -m torch.distributed.run --master_port 13919 --nproc_per_node=8 train.py --git-commit --cfg-path lavis/projects/blip/train/vrd_concate_ft_xdecoder.yaml --job-name $1

# --git-commit