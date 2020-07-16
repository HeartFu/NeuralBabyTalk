import json
import os

import torch
import numpy as np
from PIL import Image
from torch.utils import data as data
from pycocotools.coco import COCO
import torchvision.transforms as transforms
# from misc import utils


class COCODataset(data.Dataset):

    def __init__(self, image_path, ann_path, split_json_input, image_size, split='train'):
        self.image_path = image_path
        self.ann_path = ann_path
        ann_path_train = os.path.join(ann_path, 'instances_train2014.json')
        ann_path_val = os.path.join(ann_path, 'instances_val2014.json')
        self.coco_train = COCO(ann_path_train)
        self.coco_val = COCO(ann_path_val)

        self.split = split
        self.split_json_input = split_json_input
        self.info = json.load(open(split_json_input))
        self.image_size = image_size

        # separate out indexes for each of the provided splits
        self.split_ix = []
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == split:
                self.split_ix.append(ix)
        print('assigned %d images to split %s' % (len(self.split_ix), split))

        # transform
        self.Resize = transforms.Resize((self.image_size, self.image_size))
        self.ToTensor = transforms.ToTensor()
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # self.transform = transform

    def resize_bbox(self, bbox, width, height, rwidth, rheight):
        """
        resize the bbox from height width to rheight rwidth
        bbox: x,y,width, height.
        """
        width_ratio = rwidth / float(width)
        height_ratio = rheight / float(height)

        # if len(bbox) != 0:
        bbox[:, 0] = bbox[:, 0] * width_ratio
        bbox[:, 2] = bbox[:, 2] * width_ratio
        bbox[:, 1] = bbox[:, 1] * height_ratio
        bbox[:, 3] = bbox[:, 3] * height_ratio

        return bbox

    def __getitem__(self, index):
        ix = self.split_ix[index]

        # load image here.
        image_id = self.info['images'][ix]['cocoid']
        file_path = self.info['images'][ix]['filepath']
        file_name = self.info['images'][ix]['filename']
        #
        # import pdb
        # pdb.set_trace()
        # coco_split = file_path.split('/')[0]
        # get the ground truth bounding box.
        if file_path == 'train2014':
            coco = self.coco_train
        else:
            coco = self.coco_val

        bbox_ann_ids = coco.getAnnIds(imgIds=image_id)
        bbox_ann = [{'label': i['category_id'], 'bbox': i['bbox']} for i in coco.loadAnns(bbox_ann_ids)]

        gt_bboxs = np.zeros((len(bbox_ann), 4))
        gt_targets = np.zeros((len(bbox_ann)))
        for i, bbox in enumerate(bbox_ann):
            gt_bboxs[i, :4] = bbox['bbox']
            gt_targets[i] = bbox['label']

        # convert from x,y,w,h to x_min, y_min, x_max, y_max
        gt_bboxs[:, 2] = gt_bboxs[:, 2] + gt_bboxs[:, 0]
        gt_bboxs[:, 3] = gt_bboxs[:, 3] + gt_bboxs[:, 1]

        # load image
        img = Image.open(os.path.join(self.image_path, file_path, file_name)).convert('RGB')

        width, height = img.size
        # resize the image.
        img = self.Resize(img)

        gt_bboxs = self.resize_bbox(gt_bboxs, width, height, self.image_size, self.image_size)

        img = self.ToTensor(img)
        img = self.res_Normalize(img)

        targets={}
        targets['labels'] = torch.tensor(gt_targets, dtype=torch.int64)
        targets['boxes'] = torch.tensor(gt_bboxs)
        targets['image_id'] = torch.tensor(image_id)
        # targets = torch.tensor(targets)
        return img, targets

    def __len__(self):
        return len(self.split_ix)
