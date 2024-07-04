# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import math

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from .box_ops import box_xyxy_to_cxcywh
from .misc import interpolate
import numpy as np


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "polygons" in target:
        polygons = target["polygons"]
        num_polygons = polygons.shape[0]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        start_coord = torch.cat([torch.tensor([j, i], dtype=torch.float32)
                                 for _ in range(polygons.shape[1] // 2)], dim=0)
        cropped_boxes = polygons - start_coord
        cropped_boxes = torch.min(cropped_boxes.reshape(num_polygons, -1, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["polygons"] = cropped_boxes.reshape(num_polygons, -1)
        fields.append("polygons")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "polygons" in target:
        polygons = target["polygons"]
        num_polygons = polygons.shape[0]
        polygons = polygons.reshape(num_polygons, -1, 2) * torch.as_tensor([-1, 1]) + torch.as_tensor([w, 0])
        target["polygons"] = polygons

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    if target is None:
        return rescaled_image, None

    # ratios = tuple(float(s) / float(s_orig) 
    #                for s, s_orig in zip(rescaled_image.size[::-1], image.size[::-1]))
    # import ipdb; ipdb.set_trace()
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "polygons" in target:
        polygons = target["polygons"]
        scaled_ratio = torch.cat([torch.tensor([ratio_width, ratio_height])
                                 for _ in range(polygons.shape[1] // 2)], dim=0)
        scaled_polygons = polygons * scaled_ratio
        target["polygons"] = scaled_polygons

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True):
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str



class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):

        from PIL import Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        # import ipdb; ipdb.set_trace()
        h, w = image.shape[1:]
        target["size"] = image.shape[1:]
        # h, w = target["size"][0], target["size"][1]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = box_xyxy_to_cxcywh(boxes)
            target["boxes"] = boxes
        if "polygons" in target:
            polygons = target["polygons"]
            scale = torch.cat([torch.tensor([w, h], dtype=torch.float32)
                               for _ in range(polygons.shape[1] // 2)], dim=0)
            polygons = polygons / scale
            target["polygons"] = polygons
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()
        target["size"] = scaled_size

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        if "masks" in target:
            masks = target['masks']
            masks = interpolate(
                masks[:, None].float(), scaled_size, mode="nearest")[:, 0] > 0.5
            target['masks'] = masks
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ["labels", "area", "iscrowd"]

        target = target.copy()
        target["size"] = torch.tensor([h, w])

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area
            fields.append("boxes")

        if "masks" in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append("masks")

        # remove elements for which the boxes or masks that have zero area
        if "boxes" in target or "masks" in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                target[field] = target[field][keep]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])

        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            padding = out_desired_size - scaled_size
            output_image = F.pad(scaled_image, [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target



class RandomSizeCropDet(object):
    def __init__(self, crop_ration=0.8, max_try: int=50):
        self.crop_ration = crop_ration
        self.max_try  = max_try
        self.bbox_clip_border = True

    def _rand_offset(self, margin):
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w


    def _crop_data(self, results: dict, crop_size):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['image']
        margin_h = max(img.size[0] - crop_size[0], 0)
        margin_w = max(img.size[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        print(img.size)
        print((crop_x1, crop_y1, crop_x2, crop_y2))

        # crop the image
        img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        img_shape = img.size
        print("cropped", img.size)

        results['image'] = img
        results['size'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('boxes', None) is not None:
            bboxes = results['boxes'].numpy()
            det_labels = results['det_labels'].numpy()
            bboxes = self.translate_(torch.from_numpy(bboxes), [-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes = self.clip_(bboxes, img_shape[:2])
            bboxes = bboxes.numpy()
            valid_inds = self.is_inside(bboxes, img_shape[:2])
            valid_inds_set = valid_inds.nonzero()[0]
            valid_rel_trp = [None,]
            if results.get('rel_tripets') is not None:
                valid_rel_trp = []
                rel_tripets = results.get('rel_tripets').numpy()
                for trp_idx in rel_tripets:
                    if (trp_idx[0] == valid_inds_set).any()  and  (trp_idx[1] == valid_inds_set).any():
                        valid_rel_trp.append(trp_idx)

                abs2rel = {r:i for i, r in enumerate(valid_inds_set)}
                rel_rel_trp = []
                for each_r in valid_rel_trp:
                    trps = [abs2rel[each_r[0]], abs2rel[each_r[1]], each_r[2]]
                    rel_rel_trp.append(trps)
                valid_rel_trp = np.array(rel_rel_trp, dtype=int)

            # If the crop does not contain any gt-bbox area, skip this image.
            if (not valid_inds.any() or len(valid_rel_trp) < 1):
                return None
            
            results['boxes'] = torch.from_numpy(bboxes[valid_inds])
            results['det_labels'] = torch.from_numpy(det_labels[valid_inds])
            results['labels'] = torch.from_numpy(det_labels[valid_inds])
            if valid_rel_trp[0] is not None:
                results['rel_tripets'] = torch.from_numpy(valid_rel_trp)

        # # crop semantic seg
        # if results.get('gt_seg_map', None) is not None:
        #     results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
        #                                                   crop_x1:crop_x2]

        return results

    def clip_(self, boxes, img_shape):
        """Clip boxes according to the image shape in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        boxes[..., 0::2] = boxes[..., 0::2].clamp(0, img_shape[1])
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0, img_shape[0])
        return boxes

    def is_inside(self,
                  boxes,
                  img_shape,
                  all_inside: bool = False,
                  allowed_border: int = 0) :
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 4),
            the output has shape (m, n).
        """
        img_h, img_w = img_shape
        if all_inside:
            return (boxes[:, 0] >= -allowed_border) & \
                (boxes[:, 1] >= -allowed_border) & \
                (boxes[:, 2] < img_w + allowed_border) & \
                (boxes[:, 3] < img_h + allowed_border)
        else:
            return (boxes[..., 0] < img_w + allowed_border) & \
                (boxes[..., 1] < img_h + allowed_border) & \
                (boxes[..., 2] > -allowed_border) & \
                (boxes[..., 3] > -allowed_border)


    def translate_(self, boxes, distances) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        assert len(distances) == 2
        return boxes + boxes.new_tensor(distances).repeat(2)


    def __call__(self,  image, results):
        
        h, w = image.size
        crop_size = np.asarray((self.crop_ration, self.crop_ration), dtype=np.float32)
        crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
        crop_size =  (int(h * crop_h + 0.5), int(w * crop_w + 0.5))

        results['image']  = image

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            aug_res = self._crop_data(results, crop_size)
            if aug_res is not None:
                image = aug_res.pop('image')
                return image, aug_res
            
        image = results.pop('image')
        return image, results

class RandomTranslatePad:
    def __init__(self, size=384, aug_translate=True):
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, image, results):

        # import ipdb; ipdb.set_trace()
        h, w = image.size[::-1]
        dw = self.size - w
        dh = self.size - h
        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        out_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        # import ipdb; ipdb.set_trace()
        out_img[top:top+h, left:left+w, :] = np.array(image)
        out_img = Image.fromarray(out_img)

        # translate box
        if 'boxes' in results.keys():
            for i in range(len(results['boxes'])):
                box = results['boxes'][i]
                box[0], box[2] = box[0]+left, box[2]+left
                box[1], box[3] = box[1]+top, box[3]+top
                results['boxes'][i] = box

        return out_img, results
