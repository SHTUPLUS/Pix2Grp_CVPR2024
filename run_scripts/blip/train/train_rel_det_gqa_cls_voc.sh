python -m torch.distributed.run --master_port 13149 --nproc_per_node=1 train.py --git-commit --cfg-path lavis/projects/blip/train/vrd_gqa_ft_cls_voc.yaml --job-name $1

