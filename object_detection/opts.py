import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--image_path', type=str, default='/home/fanfu/newdisk/dataset/coco/2014',
                        help='path to coco image data')
    parser.add_argument('--ann_path', type=str, default='../data/coco/annotations',
                        help='path to annotation files of coco dataset')
    parser.add_argument('--split_json_input', type=str, default='../data/coco/dataset_coco.json',
                        help='file to split the train/val/test dataset')
    parser.add_argument('--image_size', type=int, default=576,
                        help='the image size')
    parser.add_argument('--num_workers', dest='num_workers',
                    help='number of worker to load data',
                    default=4, type=int)
    parser.add_argument('--cuda', type=bool, default=True,
                    help='whether use cuda')
    parser.add_argument('--mGPUs', type=bool, default=False,
                    help='whether use multiple GPUs')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=50,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--lr_decay_step', type=int, default=3,
                    help='how many steps the learning rate will decay')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float,
                    help='learning rate decay ratio')


    # Optimization: for the CNN
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? sgd|adam')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='cnn learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                    help='cnn alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--momentum', type=float, default=0.1)

    # Evaluation/Checkpointing
    parser.add_argument('--val_split', type=str, default='val',
                    help='')
    parser.add_argument('--val_images_use', type=int, default=5000,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--val_every_epoch', type=int, default=3,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--checkpoint_path', type=str, default='object_detection/save',
                    help='directory to store checkpointed models')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    # Training
    parser.add_argument('--TRAIN_DOUBLE_BIAS', type=bool, default=False,
                        help='')
    parser.add_argument('--TRAIN_WEIGHT_DECAY', type=float, default=0.0001,
                        help='')

    args = parser.parse_args()

    return args