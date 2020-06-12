import time

import torch
from torch.autograd import Variable
from tqdm import tqdm
from misc import utils


def eval_NBT(opt, model, dataset_val):
    model.eval()
    #########################################################################################
    # eval begins here
    #########################################################################################
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.num_workers)
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

    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []
    progress_bar = tqdm(dataloader_val, desc='|Validation process', leave=False)
    # for step in range(len(dataloader_val)):
    for step, data in enumerate(progress_bar):
        # data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id = data

        proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]

        # FF: Fix the bug with .data not run in the Pytorch
        input_imgs.resize_(img.size()).copy_(img)
        input_seqs.resize_(iseq.size()).copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.resize_(num.size()).copy_(num)
        input_ppls.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).copy_(bboxs)
        # FF: modify 0/1 to true/false
        mask_bboxs.resize_(box_mask.size()).copy_(box_mask.bool())
        # mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.resize_(img.size()).copy_(img)

        eval_opt = {'sample_max': 1, 'beam_size': opt.beam_size, 'inference_mode': True, 'tag_size': opt.cbs_tag_size}
        seq, bn_seq, fg_seq = model(input_imgs, input_seqs, gt_seqs, \
                                    input_num, input_ppls, gt_bboxs, mask_bboxs, 'sample', eval_opt)

        sents = utils.decode_sequence(dataset_val.itow, dataset_val.itod, dataset_val.ltow, dataset_val.itoc, dataset_val.wtod, \
                                      seq.data, bn_seq.data, fg_seq.data, opt.vocab_size, opt)
        for k, sent in enumerate(sents):
            entry = {'image_id': img_id[k].item(), 'caption': sent}
            predictions.append(entry)
            if num_show < 20:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                num_show += 1


    print('Total image to be evaluated %d' % (len(predictions)))
    lang_stats = None
    if opt.language_eval == 1:
        if opt.decode_noc:
            lang_stats = utils.noc_eval(predictions, str(1), opt.val_split, opt)
        else:
            lang_stats = utils.language_eval(opt.dataset, predictions, str(1), opt.val_split, opt)

    print('Saving the predictions')

    # Write validation result into summary
    # if tf is not None:
    #     for k, v in lang_stats.items():
    #         add_summary_value(tf_summary_writer, k, v, iteration)
    #     tf_summary_writer.flush()

    # TODO: change the train process
    # val_result_history[iteration] = {'lang_stats': lang_stats, 'predictions': predictions}
    # if wandb is not None:
    #     wandb.log({k: v for k, v in lang_stats.items()})
    return lang_stats, predictions

