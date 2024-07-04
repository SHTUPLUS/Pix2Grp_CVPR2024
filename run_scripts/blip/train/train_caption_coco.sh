python -m torch.distributed.run  --master_port 13455 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/caption_coco_ft.yaml --job-name $1
