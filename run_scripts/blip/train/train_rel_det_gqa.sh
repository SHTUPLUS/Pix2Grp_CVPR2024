python -m torch.distributed.run --master_port 13666 --nproc_per_node=8 train.py  --cfg-path lavis/projects/blip/train/vrd_gqa_ft.yaml --job-name $1
