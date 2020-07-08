from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import torch.backends.cudnn as cudnn

import opts
from eval_tools import eval_NBT
from misc import utils, AttModel
import yaml

# python evaluation.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 4 --max_epoch 30 --beam_size 3 --start_from save/normal_coco_1024
if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    cudnn.benchmark = True

    if opt.dataset == 'flickr30k':
        from misc.dataloader_flickr30k import DataLoader
    else:
        from misc.dataloader_coco import DataLoader

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset_val = DataLoader(opt, split=opt.val_split)
    # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
    #                                              shuffle=False, num_workers=opt.num_workers)

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)

    ####################################################################################
    # Build the Model
    ####################################################################################
    opt.vocab_size = dataset_val.vocab_size
    opt.detect_size = dataset_val.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset_val.fg_size
    opt.fg_mask = torch.from_numpy(dataset_val.fg_mask).byte()
    opt.glove_fg = torch.from_numpy(dataset_val.glove_fg).float()
    opt.glove_clss = torch.from_numpy(dataset_val.glove_clss).float()
    opt.glove_w = torch.from_numpy(dataset_val.glove_w).float()
    opt.st2towidx = torch.from_numpy(dataset_val.st2towidx).long()

    opt.itow = dataset_val.itow
    opt.itod = dataset_val.itod
    opt.ltow = dataset_val.ltow
    opt.itoc = dataset_val.itoc

    # choose the attention model
    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    else:
        model = AttModel.Att2in2Model(opt)

    if opt.start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')

        # opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model weights, path is %s...' % (model_path))
        model.load_state_dict(torch.load(model_path))

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    ####################################################################################
    # Evaluate the model
    ####################################################################################
    lang_stats, predictions = eval_NBT(opt, model, dataset_val)

    print('print the evaluation:')
    for k, v in lang_stats.items():
        print('{}:{}'.format(k, v))

    print('predictions: {}'.format(predictions))
    # print({k: v for k, v in lang_stats.items()})
