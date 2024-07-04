python -m torch.distributed.run --master_port 13969 --nproc_per_node=8 train.py  --cfg-path lavis/projects/blip/train/vrd_vg_ft.yaml --job-name $1

# --git-commit