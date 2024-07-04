# python train.py --cfg-path lavis/projects/blip/train/vrd_oiv6_ft_cls_voc.yaml --job-name "$1"

python -m torch.distributed.run --master_port 13919 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip/train/vrd_oiv6_ft_cls_voc.yaml --job-name $1