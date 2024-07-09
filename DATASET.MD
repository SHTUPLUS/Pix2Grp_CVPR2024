# DATASET
We adopt the same protocol with BGNN for Visual Genome and Openimage datasets.

## Visual Genome
The following is adapted from BGNN by following the same protocal of [Unbiased Scene Graph Generation from Biased Training](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) 
You can download the annotation directly by following steps.

1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `/path/to/vg/VG_100K_images`. 

2. Download the [scene graphs annotations](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=N8gWIS) and extract them to `/path/to/vg/vg_motif_anno`.

3. Move the categories_dict.json to dataset directory: 
```bash
mv all_categories_dict\vg\vg_motif_anno\categories_dict.json /path/to/vg/vg_motif_anno
```

4. Move or link the image and annotation into the project folder:
```bash
ln -s /path-to-vg_datasets/vg cache
``` 

5. Create a soft link for open-vocabulary SGG training:
``` bash
ln -s cache/vg/vg_motif_anno/VG-SGG-train.h5 cache/vg/vg_motif_anno/VG-SGG-train_zs_pred.h5
``` 
Unseen classes will be filtered during dataset loading according to the filename with keyword: `zs_pred`.


## PSG
1. Access the initial dataset from [OpenPSG](https://github.com/Jingkang50/OpenPSG/).

2. In our study, we partition the training set into validation and training subsets, and reserve a portion of the validation set for testing purposes. Download the split data from TODO

3. Move or link the image and annotation into the project folder, include images (COCO2017) and annotation:
```bash
ln -s /path-to-vg_datasets/psg cache
ln -s /path-to-coco2017 cache/coco # include train2017 and val2017
``` 

4. Move the categories_dict.json to dataset directory: 
```bash
mv all_categories_dict\psg\categories_dict.json /path-to-vg_datasets/psg
```


5. Create a soft link for open-vocabulary SGG training:
``` bash
ln -s cache/psg/psg_train.json cache/psg/psg_train_zs_pred.json
``` 


## Openimage V6 
We adopt Openimage datasets from BGNN.
1. The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).

2. The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 

3. You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3)
3. By unzip the downloaded datasets, the dataset dir contains the `images` and `annotations` folder. 
Link the `open-imagev6` dir to the `./cache/openimages` then you are ready to go.
```bash
mkdir datasets/openimages
ln -s /path/to/open_imagev6 datasets/openimages ./cache/cache
```