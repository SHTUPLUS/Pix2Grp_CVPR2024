python -m torch.distributed.run --master_port 13969 --nproc_per_node=$2 train.py --cfg-path lavis/projects/blip/train/vrd_vg_ft_xdecoder_zspred.yaml --job-name $1

