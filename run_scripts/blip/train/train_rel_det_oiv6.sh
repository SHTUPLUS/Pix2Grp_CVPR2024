python -m torch.distributed.run --master_port 13991 --nproc_per_node=4 train.py  --cfg-path lavis/projects/blip/train/vrd_oiv6_ft.yaml --job-name $1

# --git-commit