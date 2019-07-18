import os
import numpy as np
import glob
import torchvision.transforms as transforms
from PIL import Image
import torch
from encoder import DataEncoder

class VESSELBboxDataset:
    
    def __init__(self, split='trainval'):

        
        data_dir = "/media/nasir/Drive1/datasets/SAR/SAR-Ship-Dataset"
        paths = glob.glob(f'{data_dir}/JPEGImages/*.jpg')
        
        ids = [os.path.splitext(os.path.basename(x))[0] for x in paths]
        if split == 'trainval':
            self.ids = ids[0: 40000]
        else:
            self.ids = ids[40000:]
        self.input_size = 256
        self.encoder = DataEncoder()

        self.data_dir = data_dir
        self.label_names = ['ship']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])

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
        
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = Image.open(img_file).convert('RGB')
        img = self.transform(img)
        annot = self.load_annotations(self.extract_boxes(anno_file))
        return {'img': img, 'annot': annot}
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

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)



