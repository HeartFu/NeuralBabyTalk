import os
import pickle

import torch
from torch import nn as nn
from torch.utils import data
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from object_detection.cocodataset.cocoDataset import COCODataset
from object_detection.evaluation.eval_pascal import eval_detection
from object_detection.model.detection.faster_rcnn import fasterrcnn_resnet101_fpn
import object_detection.opts as opts


# import wandb

def collate_fn(batch):
    return tuple(zip(*batch))


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def trainer(opts):
    print('---------------- start train process -------------------')

    # build dataloader
    train_dataset = COCODataset(opts.image_path, opts.ann_path,
                                opts.split_json_input, opts.image_size, split='train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=opts.batch_size,
                                       shuffle=True, num_workers=opts.num_workers, collate_fn=collate_fn)

    val_dataset = COCODataset(opts.image_path, opts.ann_path,
                              opts.split_json_input, opts.image_size, split='val')
    val_dataloader = data.DataLoader(val_dataset, batch_size=opts.batch_size,
                                     shuffle=False, num_workers=opts.num_workers, collate_fn=collate_fn)

    # build the model
    model = fasterrcnn_resnet101_fpn(pretrained=False, trainable_backbone_layers=0)
    print("model structure: {}".format(model))
    infos = {}
    # optimizer
    params = []
    lr = opts.lr
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            print(key)
            if 'bias' in key:
                params += [{'params': [value], 'lr': opts.lr * (opts.TRAIN_DOUBLE_BIAS + 1),
                            'weight_decay': opts.TRAIN_BIAS_DECAY and opts.TRAIN_WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': opts.lr, 'weight_decay': opts.TRAIN_WEIGHT_DECAY}]

    if opts.optim == 'adam':
        lr = opts.lr * 0.1
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.SGD(params, momentum=opts.momentum)

    if opts.mGPUs:
        model = nn.DataParallel(model)

    if opts.cuda:
        model.cuda()

    loss_list = []
    map_list = []
    lr_list = []
    best_val_result = 0
    for epoch in range(opts.epochs):
        model.train()
        loss_temp = 0
        # start = time.time()

        if (epoch + 1) % (opts.lr_decay_step) == 0:
            adjust_learning_rate(optimizer, opts.lr_decay_gamma)
            lr *= opts.lr_decay_gamma

        # Start to training one epoch
        progress_bar = tqdm(train_dataloader, desc='|Train Epoch {}'.format(epoch), leave=False)
        for step, batch in enumerate(progress_bar):
            imgs, targets = batch
            image_list = []
            target_list = []
            for i in range(0, len(targets)):
                # delete some empty data
                if bool(targets[i]['labels'].numel()) and bool(targets[i]['boxes'].numel()):
                    target_list.append(targets[i])
                    image_list.append(imgs[i])
                else:
                    continue
            target_tuple = tuple(target_list)
            image_tuple = tuple(image_list)
            if len(target_tuple) > 0:
                if opts.cuda:
                    image_list = list(image.cuda() for image in image_tuple)
                    target_list = [{k: v.cuda() for k, v in t.items()} for t in target_tuple]
                # if opts.cuda:
                #     imgs.cuda()
                #     targets.cuda()

                model.zero_grad()
                # import pdb
                # pdb.set_trace()
                _, loss = model(image_list, target_list)

                total_loss = 0
                for k in loss.keys():
                    total_loss += loss[k].mean()
                loss_temp += total_loss.item()
                total_loss.backward()
                optimizer.step()
                description = {
                    k: loss[k].item() for k in loss.keys()
                }
                description['total_loss'] = total_loss.item()
                # description['id'] = target_list[0]['image_id']
                # wandb.log(description)
                progress_bar.set_postfix(description,
                                         refresh=True)
        result = None
        if (epoch + 1) % opts.val_every_epoch == 0:
            # evaluation, use pascal voc evaluation method, for COCO evaluation, It is used after training
            model.eval()
            pred_bboxes, pred_labels, pred_scores = list(), list(), list()
            gt_bboxes, gt_labels = list(), list()
            progress_bar_val = tqdm(val_dataloader, desc='|Val Epoch {}'.format(epoch), leave=False)
            for step, batch in enumerate(progress_bar_val):
                imgs, targets = batch

                if opts.cuda:
                    imgs.cuda()
                    targets.cuda()

                model.zero_grad()

                _, prediction = model(imgs)

                for i in range(len(prediction)):
                    gt_bboxes += list(targets[i]['boxes'].cpu().numpy())
                    gt_labels += list(targets[i]['labels'].cpu().numpy())
                    pred_bboxes.append(prediction[i]['boxes'].cpu().numpy())
                    pred_labels.append(prediction[i]['labels'].cpu().numpy())
                    pred_scores.append(prediction[i]['scores'].cpu().numpy())

            result = eval_detection(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, None, iou_thresh=0.5,
                use_07_metric=False
            )
            # wandb.log({
            #     'map': result['map']
            # })
            model.train()

        if result is not None:
            map = result['map']
        else:
            map = -1
        if map > best_val_result:
            best_val_result = map
            # save the best weight for model
            checkpoint_path = os.path.join(opts.checkpoint_path, 'model-best.pth')
            if opts.mGPUs:
                torch.save(model.module.state_dict(), checkpoint_path)
                # torch.save(model.module.state_dict(), os.path.join(wandb.run.dir, 'model-best.pth'))
            else:
                torch.save(model.state_dict(), checkpoint_path)
                # torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model-best.pth'))
            print("model saved to {} with best map score {:.3f}".format(checkpoint_path, best_val_result))
        print("Train processing, Epoch: {}, loss: {}, validation_map: {}".format(epoch, loss_temp, map))
        loss_list.append(loss_temp)
        lr_list.append(lr)
        map_list.append(map)

    infos['best_val_score'] = best_val_result
    infos['opt'] = opts
    infos['loss_history'] = loss_list
    infos['lr_history'] = lr_list
    infos['map_history'] = map_list

    with open(os.path.join(opts.checkpoint_path, 'infos_' + '-best.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    # save final model weights
    checkpoint_path = os.path.join(opts.checkpoint_path, 'model.pth')
    if opts.mGPUs:
        torch.save(model.module.state_dict(), checkpoint_path)
        # torch.save(model.module.state_dict(), os.path.join(wandb.run.dir, 'model.pth'))
    else:
        torch.save(model.state_dict(), checkpoint_path)
        # torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pth'))


# python main_obj_det.py --image_path /import/nobackup_mmv_ioannisp/shared/datasets/coco2014/images --batch_size 64 --num_workers 10 --mGPUs True
if __name__ == '__main__':
    opts = opts.parse_opt()
    print(opts)
    cudnn.benchmark = True
    # wandb.init(project="fasterrcnn")
    trainer(opts)
