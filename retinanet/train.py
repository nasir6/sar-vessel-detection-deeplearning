import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import VESSELBboxDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import voc_eval
# import csv_eval
assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
	parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
	parser = parser.parse_args(args)

	# Create the data loaders
	dataset_train = VESSELBboxDataset()
	dataset_val = VESSELBboxDataset(split='test')
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater)
	dataloader_train = DataLoader(dataset_train, num_workers=1, batch_size=8, collate_fn=collater)


	def print_inline(line):
		sys.stdout.write(f'\r{line}')
		sys.stdout.flush()
		
	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=1, pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=1, pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=1, pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=1, pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=1, pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True
	best_map = float('-inf')

	if use_gpu:
		retinanet = retinanet.cuda()
	if parser.resume:
		print_inline('==> Resuming from checkpoint..\n\n')
		# checkpoint = torch.load('./checkpoint/ckpt.pth')
		# retinanet.load_state_dict(checkpoint['net'])
		best_map = np.load('best_map.npy').item()
		print_inline(f'\n\n{best_map}')
		retinanet = torch.load('./checkpoint/ckpt.pt')
		
	
	retinanet = retinanet.cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()

		epoch_loss = []
		print_inline('\n\n')
		for iter_num, data in enumerate(dataloader_train):
			try:
				optimizer.zero_grad()

				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss

				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				print_inline('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		eval_result = voc_eval.evaluate_voc(dataset_val, retinanet)
		print_inline(f' \n map: ----->    {eval_result["map"]} \n ')
		
		if eval_result['map'] > best_map:
			print_inline(f'Saving .... \n ')
			torch.save(retinanet, 'checkpoint/ckpt.pt')
			best_map = eval_result['map']
			np.save('best_map', np.array(best_map))

		scheduler.step(np.mean(epoch_loss))

	retinanet.eval()

if __name__ == '__main__':
	main()
