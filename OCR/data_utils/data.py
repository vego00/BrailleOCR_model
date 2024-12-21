#!/usr/bin/env python
# coding: utf-8

import os
import random
import json
import PIL
import numpy as np
import albumentations
import albumentations.augmentations.transforms as T
import albumentations.augmentations.functional as albu_f
import torch
import cv2

from braille_utils import label_tools as lt
import local_config


def rect_vflip(b):
    return b[:4] + (lt.label_vflip(b[4]),)

def rect_hflip(b):
    return b[:4] + (lt.label_hflip(b[4]),)


def common_aug(mode, params):
    augs_list = []
    assert mode  in {'train', 'debug', 'inference'}
    if mode == 'train':
        augs_list.append(albumentations.PadIfNeeded(
            min_height=params.data.net_hw[0],
            min_width=params.data.net_hw[1],
            border_mode=cv2.BORDER_REPLICATE,
            always_apply=True))
        augs_list.append(albumentations.RandomCrop(
            height=params.data.net_hw[0],
            width=params.data.net_hw[1],
            always_apply=True))
    elif mode == 'debug':
        augs_list.append(albumentations.CenterCrop(
            height=params.data.net_hw[0],
            width=params.data.net_hw[1],
            always_apply=True))
    if mode != 'inference':
        if params.augmentation.get('blur_limit', 4):
            augs_list.append(T.Blur(blur_limit=params.augmentation.get('blur_limit', 4)))
        if params.augmentation.get('RandomBrightnessContrast', True):
            augs_list.append(T.RandomBrightnessContrast())
        if params.augmentation.get('JpegCompression', True):
            augs_list.append(T.JpegCompression(quality_lower=30, quality_upper=100))
        if params.augmentation.get('HorizontalFlip', True):
            augs_list.append(T.HorizontalFlip())

    return albumentations.ReplayCompose(augs_list, p=1., bbox_params = {'format':'albumentations', 'min_visibility':0.5})


class ImagePreprocessor:
    def __init__(self, params, mode):
        assert mode in {'train', 'debug', 'inference'}
        self.params = params
        self.albumentations = common_aug(mode, params)

    def preprocess_and_augment(self, img, rects=[]):
        aug_img = self.random_resize_and_stretch(img,
                                                 new_width_range=self.params.augmentation.img_width_range,
                                                 stretch_limit=self.params.augmentation.stretch_limit)
        aug_res = self.albumentations(image=aug_img, bboxes=rects)
        aug_img = aug_res['image']
        aug_bboxes = aug_res['bboxes']
        aug_bboxes = [b for b in aug_bboxes if
                      b[0] > 0 and b[0] < 1 and
                      b[1] > 0 and b[1] < 1 and
                      b[2] > 0 and b[2] < 1 and
                      b[3] > 0 and b[3] < 1]
        if not self.params.data.get('get_points', False):
            for t in aug_res['replay']['transforms']:
                if t['__class_fullname__'].endswith('.VerticalFlip') and t['applied']:
                    aug_bboxes = [rect_vflip(b) for b in aug_bboxes]
                if t['__class_fullname__'].endswith('.HorizontalFlip') and t['applied']:
                    aug_bboxes = [rect_hflip(b) for b in aug_bboxes]
        return aug_img, aug_bboxes

    def random_resize_and_stretch(self, img, new_width_range, stretch_limit = 0):
        new_width_range = T.to_tuple(new_width_range)
        stretch_limit = T.to_tuple(stretch_limit, bias=1)
        new_sz = int(random.uniform(new_width_range[0], new_width_range[1]))
        stretch = random.uniform(stretch_limit[0], stretch_limit[1])
        
        img_max_sz = img.shape[1]
        new_width = int(img.shape[1]*new_sz/img_max_sz)
        new_width = ((new_width+31)//32)*32
        new_height = int(img.shape[0]*stretch*new_sz/img_max_sz)
        new_height = ((new_height+31)//32)*32
        return albu_f.resize(img, new_height, new_width, interpolation=cv2.INTER_LINEAR)

    def to_normalized_tensor(self, img, device='cpu'):
        assert img.ndim == 3
        ten_img = torch.from_numpy(img.transpose((2, 0, 1))).to(device).float()
        means = ten_img.view(3, -1).mean(dim=1)
        std = torch.max(ten_img.view(3, -1).std(dim=1), torch.tensor(self.params.data.get('max_std',0)*255).to(ten_img))
        ten_img = (ten_img - means.view(-1, 1, 1)) / (3*std.view(-1, 1, 1))
        ten_img = ten_img.mean(dim=0).expand(3, -1, -1)
        return ten_img


def unify_shape(img):
    if len(img.shape) == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


class BrailleDataset(torch.utils.data.ConcatDataset):
    def __init__(self, params, list_file_names, mode, verbose):
        sub_datasets = []
        for list_file_name in list_file_names:
            if isinstance(list_file_name, (tuple, list)):
                list_params = list_file_name[2] if len(list_file_name) >=3 else {}
                assert isinstance(list_params, dict)
                list_file_name, sample_weight = list_file_name[:2]
            else:
                sample_weight = 1
                list_params = {}
            while sample_weight >= 1:
                sub_datasets.append(BrailleSubDataset(params, list_file_name, mode, verbose, 1, list_params))
                sample_weight -= 1
            if sample_weight > 1e-10:
                sub_datasets.append(BrailleSubDataset(params, list_file_name, mode, verbose, sample_weight, list_params))

        super(BrailleDataset, self).__init__(sub_datasets)

class BrailleSubDataset:

    def __init__(self, params, list_file_name, mode, verbose, sample_weight, list_params):
        assert mode in {'train', 'debug', 'inference'}
        self.params = params
        self.mode = mode
        self.image_preprocessor = ImagePreprocessor(params, mode)

        self.image_files = []
        self.label_files = []

        list_file = os.path.join(local_config.data_path, list_file_name)
        data_dir = os.path.dirname(list_file)
        with open(list_file, 'r') as f:
            files = f.readlines()
        for fn in files:
            if fn[-1] == '\n':
                fn = fn[:-1]
            fn = fn.replace('\\', '/')
            image_fn, labels_fn = self.filenames_of_item(data_dir, fn)
            if image_fn:
                self.image_files.append(image_fn)
                self.label_files.append(labels_fn)
            else:
                print("WARNING: can't load file:", data_dir, fn)

        assert len(self.image_files) > 0, list_file

        self.images = [None] * len(self.image_files)
        self.rects = [None] * len(self.image_files)
        self.aug_images = [None] * len(self.image_files)
        self.aug_bboxes = [None] * len(self.image_files)
        self.REPEAT_PROBABILITY = 0.6
        self.verbose = verbose
        assert sample_weight <= 1
        if sample_weight < 1:
            assert mode == 'train'
        self.denominator = int(1/sample_weight)
        self.call_count = 0
        self.list_params = list_params

    def __len__(self):
        return len(self.image_files) // self.denominator

    def __getitem__(self, item):
        if self.denominator > 1:
            self.call_count = (self.call_count + 1) % self.denominator
            item = item * self.denominator + self.call_count
        img = self.images[item]
        if img is None:
            img_fn = self.image_files[item]
            img = PIL.Image.open(img_fn)
            img = np.asarray(img)
            img= unify_shape(img)
            assert len(img.shape) == 3 and img.shape[2] == 3, (img_fn, img.shape)
            self.images[item] = img
        width = img.shape[1]
        height = img.shape[0]
        rects = self.rects[item]
        if rects is None:
            lbl_fn = self.label_files[item]
            rects = self.read_annotation(lbl_fn, width, height)
            rects = [ (min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])) + r[4:] for r in rects]
            rects = [ r for r in rects if r[0] < r[2] and r[1] < r[3]]
            self.rects[item] = rects

        if (self.aug_images[item] is not None) and (random.random() < self.REPEAT_PROBABILITY):
            aug_img = self.aug_images[item]
            aug_bboxes = self.aug_bboxes[item]
        else:
            aug_img, aug_bboxes = self.image_preprocessor.preprocess_and_augment(img, rects)
            self.aug_images[item] = aug_img
            self.aug_bboxes[item] = aug_bboxes

        if self.verbose >= 2:
            print('BrailleDataset: preparing file '+ self.image_files[item] + '. Total rects: ' + str(len(aug_bboxes)))

        if self.mode == 'train':
            return self.image_preprocessor.to_normalized_tensor(aug_img), np.asarray(aug_bboxes).reshape(-1, 5), self.list_params
        else:
            return self.image_preprocessor.to_normalized_tensor(aug_img), np.asarray(aug_bboxes).reshape(-1, 5), self.list_params, aug_img

    def filenames_of_item(self, data_dir, fn):
        def check_label_ext(image_fn, ext):
            if not os.path.isfile(image_fn):
                return None
            lbl_fn = image_fn.rsplit('.',1)[0]+ext
            if os.path.isfile(lbl_fn):
                return lbl_fn
            return None

        full_fn = os.path.join(data_dir, fn)
        lbl_fn = check_label_ext(full_fn, '.json')
        if lbl_fn:
            return full_fn, lbl_fn
        lbl_fn = check_label_ext(full_fn, '.txt')
        if lbl_fn:
            return full_fn, lbl_fn
        full_fn = full_fn.rsplit('.', 1)[0] + '+recto.jpg'
        lbl_fn = check_label_ext(full_fn, '.txt')
        if lbl_fn:
            return full_fn, lbl_fn
        return None, None

    def read_annotation(self, label_filename, width, height):
        ext = label_filename.rsplit('.', 1)[-1]
        if ext == 'json':
            return read_LabelMe_annotation(label_filename, self.params.data.get('get_points', False))
        else:
            raise ValueError("unsupported label file type: " + ext)


def limiting_scaler(source, dest):
    def scale(x):
        return int(min(max(0, x), source-1)) * dest/source
    return scale


def read_LabelMe_annotation(label_filename, get_points):
    if get_points:
        raise NotImplementedError("read_annotation get_point mode not implemented for LabelMe annotation")
    with open(label_filename, 'r', encoding='cp1251') as opened_json:
        loaded = json.load(opened_json)
    convert_x = limiting_scaler(loaded["imageWidth"], 1.0)
    convert_y = limiting_scaler(loaded["imageHeight"], 1.0)
    rects = [(convert_x(min(xvals)),
              convert_y(min(yvals)),
              convert_x(max(xvals)),
              convert_y(max(yvals)),
              lt.human_label_to_int(label),
              ) for label, xvals, yvals in
                    ((shape["label"],
                      [coords[0] for coords in shape["points"]],
                      [coords[1] for coords in shape["points"]]
                     ) for shape in loaded["shapes"]
                    )
            ]
    return rects


def create_dataloader(params, collate_fn, list_file_names, shuffle, mode = 'train', verbose = 0):
    dataset = BrailleDataset(params, list_file_names=list_file_names, mode=mode, verbose=verbose)
    loader = torch.utils.data.DataLoader(dataset, params.data.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)
    return loader
