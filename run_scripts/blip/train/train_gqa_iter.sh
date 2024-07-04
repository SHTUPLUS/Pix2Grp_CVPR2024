python -m torch.distributed.run --master_port 13566 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/gqa_ft_iter.yaml --job-name $1


# python -m torch.distributed.run --master_port 13562 --nproc_per_node=1 train.py --cfg-path lavis/projects/blip/train/gqa_ft.yaml