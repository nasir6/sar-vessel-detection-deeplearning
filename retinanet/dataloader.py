from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image
import glob 

class VESSELBboxDataset:
    
    def __init__(self, split='trainval', scale = 1):

        
        data_dir = "/media/nasir/Drive1/datasets/SAR/SAR-Ship-Dataset"
        paths = glob.glob(f'{data_dir}/JPEGImages/*.jpg')
        
        ids = [os.path.splitext(os.path.basename(x))[0] for x in paths]
        if split == 'trainval':
            self.ids = ids[0: 40000]
        else:
            self.ids = ids[40000:]
        self.input_size = 256

        self.data_dir = data_dir
        self.label_names = ['ship']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        self.split = split
        self.scale = scale

    def transform_resize(self, image, annots, scale=1):
        image = np.array(image)
        rows, cols, cns = image.shape
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        x = self.transform(image)
        # x = x.unsqueeze(0)
        annots_scaled = (np.array(annots)*scale).astype(int)
        return x, annots_scaled
        
    def __len__(self):
        return len(self.ids)

    def str2int(self, a):
        return [int(x) for x in a]

    def extract_boxes(self, fname):
        with open(fname) as f:
            content = f.readlines()
            f.close()
            content = [x.strip() for x in content]
            content = [self.str2int(x.split(' ')[-4:]) for x in content]
            return content

    def __getitem__(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno_file = os.path.join(self.data_dir, 'ground-truth', id_ + '.txt')
        # bbox = self.extract_boxes(anno_file)
        
        # label = list()
        
        
        # bbox = np.stack(bbox).astype(np.float32)
        # bb = np.ones_like(bbox).astype(np.float32)
        # for i in range(len(bbox)):
        #     label.append(0)

        # bb[:, 0] = bbox[:, 1]
        # bb[:, 1] = bbox[:, 0]
        # bb[:, 2] = bbox[:, 3] + bbox[:, 1]
        # bb[:, 3] = bbox[:, 2] + bbox[:, 0]
        # label = np.stack(label)
        annot = self.load_annotations(self.extract_boxes(anno_file))
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = Image.open(img_file).convert('RGB')
        if self.split == 'test':
            img , annot = self.transform_resize(img, annot, self.scale)
        else:
            img = self.transform(img)
        return {'img': img, 'annot':  torch.Tensor(annot)}
        # return {img, torch.Tensor(bb).type(torch.float)}

    def load_annotations(self, bboxes):
        annotations     = np.zeros((0, 5))
        if len(bboxes) == 0:
            return annotations
        for idx, box in enumerate(bboxes):
            annotation        = np.zeros((1, 5))
            annotation[0, :4] = box
            annotation[0, 4]  = 0
            annotations       = np.append(annotations, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [1 for s in data]
        
    # widths = [int(s.shape[0]) for s in imgs]
    # heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    # max_width = np.array(widths).max()
    # max_height = np.array(heights).max()
    # print(max_height, max_width, batch_size, imgs[0].shape)
    padded_imgs = torch.zeros(batch_size, 3, 256, 256)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    # padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=256, max_side=512):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
