from __future__ import print_function


import numpy as np
import os, sys
from utils_rcnn.eval_tool import eval_detection_voc

import torch
def print_inline(line):
    sys.stdout.write(f'\r{line}')
    sys.stdout.flush()
def evaluate_voc(dataset, model, threshold=0.5):
    print_inline('\n\n')
    model.eval()
    
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            # scale = data['scale']

            # run network
            scores, labels, boxes = model(data['img'].cuda().float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()


            gt_bboxes.append((data['annot'][:,:4].numpy()))
            gt_labels.append((data['annot'][:,4].numpy()))
            
            gt_difficults += [0 for x in range(data['annot'].shape[0])]
            pred_bboxes.append((np.array([box.numpy() for box in boxes])))
            pred_labels.append((np.array([lable.numpy() for lable in labels])))
            pred_scores.append((np.array([s.numpy() for s in scores])))

            print_inline('{}/{}'.format(index, len(dataset)))
        

        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, None,
            use_07_metric=True)
        

        model.train()

        return result
