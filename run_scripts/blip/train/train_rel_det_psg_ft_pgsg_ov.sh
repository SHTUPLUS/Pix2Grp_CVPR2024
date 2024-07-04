
pkill -f lets; pkill -f python

python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py --git-commit --cfg-path lavis/projects/blip/train/vrd_psg_ft_pgsg_ov.yaml --job-name $1 