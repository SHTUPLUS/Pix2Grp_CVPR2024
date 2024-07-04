from pathlib import Path
from collections import OrderedDict
import logging

import os
import os.path
import json
from collections import Counter

import h5py
from PIL import Image
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from lavis.datasets.datasets.oiv6_rel_detection import VisualRelationDetectionEvalDataset, load_categories_info, prepare_anno
import lavis.datasets.datasets.utils.transforms_vt as T

BOX_SCALE = 1024  # Scale at which we have the boxes


logger = logging.getLogger(__name__)

class VGVRDEvalDataset(VisualRelationDetectionEvalDataset):

    zs_predicate = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 23, 25, 26, 27, 31, 34, 35, 36, 40, 41, 42, 44, 48]
    # zs_predicate = [3, 5, 9, 12, 14, 16, 21, 23, 25, 27, 31, 34, 36, 40, 46]

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        ann_path = ann_paths[0]
        image_info_root = ann_paths[1]

        self.split = None
        if 'train' in ann_path:
            self.split = 'train'
        if 'val' in ann_path:
            self.split = 'val'
        if 'test' in ann_path:
            self.split = 'test'

        self.zeroshot_predicate_mode = False
        if "zs_pred" in ann_path:
            self.zeroshot_predicate_mode = True

        ann_path_dir = "/".join(ann_path.split('/')[:-1])
        (self.ind_to_entities,
         self.ind_to_predicates,
         self.entities_to_ind,
         self.predicate_to_ind) = load_categories_info(os.path.join(ann_path_dir, 'categories_dict.json'))

        (self.split_mask,
         self.gt_boxes,
         self.gt_classes,
         self.gt_attributes,
         self.relationships) = self._load_annotations(
            ann_path,
        )

        self.filenames, self.img_info = load_image_filenames(
            vis_root,
            image_info_root,
            False  # length equals to split_mask
        )
        self.filenames = [self.filenames[i]
                          for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i]
                         for i in np.where(self.split_mask)[0]]
        self.ids = list(range(len(self.filenames)))

        # self.ids = self.ids[:14000]

        self.vis_processor = vis_processor  # val processor for simple version
        self.text_processor = None

        # self.get_frequency(ann_path_dir)

        if self.zeroshot_predicate_mode:
            unseen_marker = torch.zeros(len(self.ind_to_predicates)).bool()
            for zs_pred_id in VGVRDEvalDataset.zs_predicate:
                unseen_marker[zs_pred_id] = True
            idx_list_filtered = []
            for idx in self.ids:
                filtered_rel = []
                for item in self.relationships[idx]:
                    if not unseen_marker[item[-1]]:
                        filtered_rel.append(item)

                if len(filtered_rel) != 0 :
                    idx_list_filtered.append(idx)

                filtered_rel = np.array(filtered_rel, dtype=np.int32)

                self.relationships[idx] = filtered_rel
            logger.info(f"predicate open vocabulary ON, data filtering: {len(self.ids) - len(idx_list_filtered)} ({len(self.ids)} -> {len(idx_list_filtered)}) ")
            self.ids = idx_list_filtered

    def _add_instance_ids(self, key="instance_id"):
        pass


    def get_frequency(self, ann_path_dir):
        if self.split == 'train':

            if not os.path.exists(os.path.join(ann_path_dir, 'freq_info.pth')):
                pred_cnt = Counter()
                ent_cnt = Counter()
                rel_ent_cnt = Counter()
                train_trp_cnt = Counter()
                ent_inst_cnt = []
                rel_inst_cnt = []
                for idx, each in enumerate(self.gt_classes):
                    
                    ent_inst_cnt.append(len(each))
                    rel_inst_cnt.append(self.relationships[idx])
                    for each_rel in self.relationships[idx]:
                        pred_cnt[each_rel[-1]] += 1
                        rel_ent_cnt[ each[each_rel[0]] ] += 1
                        rel_ent_cnt[ each[each_rel[1]] ] += 1
                        
                        train_trp_cnt[f"{each[each_rel[0]]}, {each[each_rel[1]]}, {each_rel[-1]}"] += 1
                    for each_ent in each:
                        ent_cnt[each_ent] += 1

                ent_rel_freqs = []
                for i in range(len(rel_ent_cnt.most_common())):
                    ent_rel_freqs.append(rel_ent_cnt[i])
                ent_rel_freqs = torch.tensor(ent_rel_freqs, dtype=torch.float)
                ent_rel_freqs = ent_rel_freqs / ent_rel_freqs.sum()

                rel_freqs = []
                for i in range(len(pred_cnt.most_common())):
                    rel_freqs.append(pred_cnt[i])
                rel_freqs = torch.tensor(rel_freqs, dtype=torch.float)
                rel_freqs = rel_freqs / rel_freqs.sum()
                
                torch.save(self.freq_info, os.path.join(ann_path_dir, 'freq_info.pth'))
        
        # self.freq_info = torch.load(os.path.join(ann_path_dir, 'freq_info.pth'))


    def collater(self, samples):
        # the final processing function for data loading
        img_tensor = default_collate([each['image'] for each in samples])
        
        anno = [each['target'] for each in samples]
        instance_id = [each['instance_id'] for each in samples]
        image_pth = [each['image_pth'] for each in samples]
        image_id = [each['image_id'] for each in samples]

        return {'image': img_tensor, 'targets': anno, "image_id": image_id,
                "instance_id": instance_id, "image_pth": image_pth,
                # 'freq_info': self.freq_info,
                }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        ids = self.ids[index]

        img_fn = self.filenames[ids]
        image_pth = os.path.join(self.vis_root, img_fn)
        image = Image.open(image_pth).convert("RGB")

        self.img_info[ids]
        w, h = image.size

        anno = {
            'img_fn': img_fn,
            'det_labels':  self.gt_classes[ids],
            'rel': self.relationships[ids],
            'bbox': self.gt_boxes[ids] / BOX_SCALE * max(w, h),
        }

        # image = Image.fromarray(np.ones((128, 128, 3), dtype=np.uint8))
        # dict_keys(['bbox', 'det_labels', 'rel', 'img_size', 'img_fn'])

        target = prepare_anno(
            image, anno, self.ind_to_entities, self.ind_to_predicates)

        image, target = self.vis_processor(image, target)

        label_texts = []
        for each_l in target['det_labels']:
            l_text = self.ind_to_entities[each_l.item()]
            label_texts.append(l_text)
        target['det_labels_texts'] = label_texts
        target["labels_texts"] = label_texts

        predc_label_texts = []
        for each_l in target['rel_tripets']:
            predc_label_texts.append(self.ind_to_predicates[each_l[-1].item()])
        target['rel_labels_texts'] = predc_label_texts
        # print([each['name'] for each in self.coco.cats.values()])
        # dict_keys('boxes': 0-1 xywh,
        # 'labels', 'image_id', 'area', 'iscrowd', 'orig_size', 'size'])
        return {
            "instance_id": ids,
            "image_id": anno['img_fn'],
            "image_pth": image_pth,
            "image": image,
            "target": target,
        }

    def _load_annotations(self, anno_file, ):
        """
        Load a json file with COCO's instances annotation format.
        Currently supports instance detection, instance segmentation,
        and person keypoints annotations.

        Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str): the directory where the images in this json file exists.
            dataset_name (str): the name of the dataset (e.g., coco_2017_train).
                If provided, this function will also put "thing_classes" into
                the metadata associated with this dataset.
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
                "category_id", "segmentation"). The values for these keys will be returned as-is.
                For example, the densepose annotations are loaded in this way.

        Returns:
            list[dict]: a list of dicts in cvpods standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )

                    Return:
            image_index: numpy array corresponding to the index of images we're using
            boxes: List where each element is a [num_gt, 4] array of ground
                        truth boxes (x1, y1, x2, y2)
            gt_classes: List where each element is a [num_gt] array of classes
            relationships: List where each element is a [num_r, 3] array of
                        (box_ind_1, box_ind_2, predicate) relationships

        Notes:
            1. This function does not read the image files.
            The results do not have the "image" field.
        """

        roi_h5 = h5py.File(anno_file, 'r')
        data_split = roi_h5['split'][:]
        split_flag = 2 if self.split == 'test' else 0
        split_mask = data_split == split_flag

        init_len = len(np.where(split_mask)[0])
        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        logger.info(f"no entities img: {init_len - len(np.where(split_mask)[0])}")

        logger.info(f"no rel img: {len(np.where(roi_h5['img_to_first_rel'][:] < 0))}")

        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

        image_index = np.where(split_mask)[0]

        # for debug
        num_im_split = 5000
        num_im = -1
        num_val_im = 5000

        if num_im > -1:
            image_index = image_index[: num_im]
        if num_val_im > 0:
            if self.split == 'val':
                image_index = image_index[: num_im_split][:num_val_im]
            else:
                image_index = image_index[num_im_split:]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = roi_h5['labels'][:, 0]
        all_attributes = roi_h5['attributes'][:, :]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, : 2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels
        _relations = roi_h5['relationships'][:]
        _relation_predicates = roi_h5['predicates'][:, 0]
        assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
        assert (_relations.shape[0]
                == _relation_predicates.shape[0])  # sanity check

        # Get everything by image.

        boxes = []
        gt_classes = []
        gt_attributes = []
        relationships = []
        for i in range(len(image_index)):

            i_obj_start = im_to_first_box[i]
            i_obj_end = im_to_last_box[i]
            i_rel_start = im_to_first_rel[i]
            i_rel_end = im_to_last_rel[i]

            boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
            # let the foreground start from 0
            gt_classes_i = all_labels[i_obj_start: i_obj_end + 1] - 1
            # the relationship and attributes foreground start from the 0
            gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :] - 1

            if i_rel_start >= 0:
                predicates = _relation_predicates[i_rel_start: i_rel_end + 1] - 1
                obj_idx = _relations[i_rel_start: i_rel_end
                                     + 1] - i_obj_start  # range is [0, num_box)
                assert np.all(obj_idx >= 0)
                assert np.all(obj_idx < boxes_i.shape[0])
                # (num_rel, 3), representing sub, obj, and pred
                rels = np.column_stack((obj_idx, predicates))
            else:
                rels = np.zeros((0, 3), dtype=np.int32)

            boxes.append(boxes_i)
            gt_classes.append(gt_classes_i)
            gt_attributes.append(gt_attributes_i)
            relationships.append(rels)

        return split_mask, boxes, gt_classes, gt_attributes, relationships


class VGVRDDataset(VGVRDEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_processor = vis_processor  # val processor for simple version
        self.text_processor = None


def load_image_filenames(img_dir, image_file, check_img_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if check_img_file:
            if os.path.exists(filename):
                fns.append(filename)
                img_info.append(img)
        else:
            fns.append(filename)
            img_info.append(img)

    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


freq_info = {'ent': [0.002568619791418314, 0.0014105848968029022, 0.005191445350646973, 0.004299512133002281, 0.00281131430529058, 0.0017801704816520214, 0.0066389888525009155, 0.006569999735802412, 0.00663652503862977, 0.007126841694116592, 0.007120681926608086, 0.005581974051892757, 0.002835953375324607, 0.0036896958481520414, 0.003100822912529111, 0.0016163209220394492, 0.003366924589499831, 0.0037340461276471615, 0.002838417189195752, 0.013355590403079987, 0.003395259380340576, 0.023829646408557892, 0.011240329593420029, 0.0016138569917529821, 0.003116838401183486, 0.009078253991901875, 0.01138200331479311, 0.007139161229133606, 0.0037204946856945753, 0.007013502065092325, 0.002858128398656845, 0.0030355295166373253, 0.004883457440882921, 0.002562460023909807, 0.0015768983867019415, 0.002875375794246793, 0.010174690745770931, 0.00733380950987339, 0.0009165722294710577, 0.011166412383317947, 0.010705661959946156, 0.002657320350408554, 0.0049031684175133705, 0.005938007961958647, 0.007708323188126087, 0.0016508155968040228, 0.0017567634349688888, 0.004988173488527536, 0.005694081541150808, 0.0026671758387237787, 0.001644655829295516, 0.010634209029376507, 0.009863006882369518, 0.00966466311365366, 0.0029874832835048437, 0.001689006108790636, 0.012638594955205917, 0.012190164066851139, 0.003573892405256629, 0.009361602365970612, 0.017816487699747086, 0.005940472241491079, 0.0027571083046495914, 0.009036366827785969, 0.0028716798406094313, 0.008226974867284298, 0.00465061841532588, 0.0019095254829153419, 0.0017814025050029159, 0.0018639431800693274, 0.0019181490642949939, 0.0044547379948198795, 0.006125264801084995, 0.015824422240257263, 0.002657320350408554, 0.005479722283780575, 0.002897550817579031, 0.09673039615154266, 0.0010114324977621436, 0.006239836569875479, 0.0026388410478830338, 0.0025895629078149796, 0.002983787562698126, 0.00555733498185873, 0.0020130095072090626, 0.0024811511393636465, 0.00844626221805811, 0.0022914304863661528, 0.0021719313226640224, 0.0067116739228367805, 0.03042551688849926, 0.0034001872409135103, 0.0036170107778161764, 0.004476913250982761, 0.008844182826578617, 0.0029345094226300716, 0.011537229642271996, 0.0037340461276471615, 0.010884295217692852, 0.0024811511393636465, 0.0019144532270729542, 0.003278224030509591, 0.001063174451701343, 0.003072488121688366, 0.0031550289131700993, 0.002014241414144635, 0.0012455034302547574, 0.0026782634668052197, 0.0033496771939098835, 0.003556645242497325, 0.0281821321696043, 0.00679544685408473, 0.006838565226644278, 0.006717833690345287, 0.013874242082238197, 0.0019218450179323554, 0.006059971638023853, 0.0035024392418563366, 0.002744788769632578, 0.0011457152431830764, 0.010327452793717384, 0.0016076972242444754, 0.0012504311744123697, 0.010260927490890026, 0.005382397677749395, 0.022377174347639084, 0.008994480594992638, 0.0029542206320911646, 0.0008697580196894705, 0.003495047567412257, 0.0022865026257932186, 0.0012393436627462506, 0.003094663145020604, 0.00787463691085577, 0.01641329564154148, 0.023488394916057587, 0.004798452835530043, 0.004171389155089855, 0.00614128028973937, 0.0043093678541481495, 0.0012036169646307826, 0.001350219245068729, 0.001960035413503647, 0.006746168714016676, 0.026424136012792587, 0.0026142019778490067, 0.004208347760140896, 0.0018269846914336085, 0.03708544746041298, 0.0076159266754984856], 'pred': [0.020723894238471985, 0.0006480066804215312, 0.0005519144469872117, 0.0012147045927122235, 0.0016729907365515828, 0.005196372978389263, 0.00390775129199028, 0.032146554440259933, 0.0016064653173089027, 0.0012590548722073436, 0.004200955852866173, 0.0011949933832511306, 0.0012615188024938107, 0.0017296604346483946, 1.2319518646108918e-05, 0.002749716630205512, 0.0004878529580309987, 0.0004237914690747857, 0.0019883704371750355, 0.17002661526203156, 0.02829054370522499, 0.060291726142168045, 0.009382545948028564, 0.0019193810876458883, 0.002527965232729912, 0.0009091804968193173, 0.00031537970062345266, 0.0006529344827868044, 0.0511481799185276, 0.08074212819337845, 0.2908318042755127, 0.0008451190078631043, 0.0031464050989598036, 0.00037697728839702904, 0.0015793623169884086, 0.001071798149496317, 0.0003301631077192724, 0.01110481470823288, 0.00012073128891643137, 0.013194205239415169, 0.006149903871119022, 0.0008056965307332575, 0.011659192852675915, 0.0014290641993284225, 0.0007120682275854051, 0.0032572809141129255, 0.002234760671854019, 0.11970137804746628, 0.012169220484793186, 0.03009658493101597]}  
