## Vessel Detection in Synthetic Aperture Radar(SAR) Images Using Deep-Learning
Deep learning models are trained on [SAR-Ship-Dataset](https://github.com/CAESAR-Radi/SAR-Ship-Dataset) to compare the detection performance between traditional vessel detection methods e.g. CFAR on SAR images and deep learning detection models.

### Dataset
An annotated dataset by SAR experts was recently(2019) published in Remote Sensing journal consisting of 43,819 ship chips is used to evaluate vessel detection "A SAR Dataset of Ship Detection for Deep Learning under Complex Backgrounds" [GitHub](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)
[Paper](https://www.mdpi.com/2072-4292/11/7/765/htm). This dataset is used to evaluate the detection. We split the dataset into training and evaluation sets. Evaluation set consists of Last 3819 images. Training set consists of first 40000 images.

### Models Trained
#### Faster RCNN
Code for Faster RCNN is cloned from [GitHub Repo](https://github.com/chenyuntc/simple-faster-rcnn-pytorch.git).
To test run demo notebook on subset of images the detection are plotted and saved in results directory.
#### Retinanet
Code for Retinanet is cloned from [GitHub Repo](https://github.com/yhenon/pytorch-retinanet.git)
To test run demo notebook on subset of images.
