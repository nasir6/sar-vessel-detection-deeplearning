import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image
import glob
from torchvision import transforms, utils
import skimage.io
import skimage.transform
import skimage.color
import skimage
from PIL import Image

class VESSELBboxDataset:
    
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False, scale=1):

        paths = glob.glob(f'{data_dir}/JPEGImages/*.jpg')
        # id_list_file = os.path.join(
        #     data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        ids = [os.path.splitext(os.path.basename(x))[0] for x in paths]
        if split == 'trainval':
            self.ids = ids[0: 40000]
        else:
            self.ids = ids[40000:]

        # [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VESSEL_BBOX_LABEL_NAMES
        self.scale = scale
        self.split = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])

    def transform_resize(self, image, annots, scale=1):
        rows, cols, cns = np.array(image).shape
        image = image.resize((int(round(rows*scale)), int(round((cols*scale)))))
        x = self.transform(image)
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

    def get_example(self, i):
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
        bbox = self.extract_boxes(anno_file)
        label = list()
        difficult = list()
        for i in range(len(bbox)):
            label.append(0)
            difficult.append(0)
        
        bbox = np.stack(bbox).astype(np.float32)
        bb = np.ones_like(bbox).astype(np.float32)
        bb[:, 0] = bbox[:, 1]
        bb[:, 1] = bbox[:, 0]
        bb[:, 2] = bbox[:, 3] + bbox[:, 1]
        bb[:, 3] = bbox[:, 2] + bbox[:, 0]
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = Image.open(img_file).convert('RGB')
        
        if self.split == 'test':
            img , bb = self.transform_resize(img, bb, self.scale)
        else:
            img = self.transform(img)
        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bb, label, difficult

    __getitem__ = get_example


VESSEL_BBOX_LABEL_NAMES = ('ship')











