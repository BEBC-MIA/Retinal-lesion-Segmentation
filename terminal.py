#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
paraser = argparse.ArgumentParser()
# control commands
paraser.add_argument('--preprocess', type=int, default=0,
                         help='1: preprocessing of original dataset, 0: skip the step')
paraser.add_argument('--vessel_seg', type=int, default=0,
                         help='1: vessel segmentation, 0: skip the step')
paraser.add_argument('--OD_seg', type=int, default=0,
                         help='1: optic disc segmentation based on vessel segmentation, 0: skip the step')
paraser.add_argument('--build_lesion_lab', type=int, default=0,
                         help='1: crop lesion and build lesion lab, 0: skip the step')
paraser.add_argument('--build_PBDA_dataset', type=int, default=0,
                         help='1: do PBDA and build an aug dataset, 0: skip the step')
paraser.add_argument('--step', type=int, default=-1, help='-1 denotes skipping the step,'
                                                         '0 denotes training and testing with random initial weight,'
                                                         '1 denotes training and testing with a fixed initial weight, '
                                                         '2 denotes testing only')

# vessel segmentation
paraser.add_argument('--vs_step', type=str, default='test', help='train, test, all')
paraser.add_argument('--vs_batch_size', type=int, default=2, help='batch size for training and testing')
paraser.add_argument('--vs_target_size', type=int, default=512, help='image size for training and testing')
paraser.add_argument('--vs_max_epoch', type=int, default=200, help='max epoch')
paraser.add_argument('--vs_root', type=str, default='vessel_seg', help='root directory of vessel seg')

# Build poisson augmentation dataset
paraser.add_argument('--dataset', type=str, default='e_ophtha',
                         help='The name of target dataset')
paraser.add_argument('--original_data_dir', type=str, default='.',
                     help='root directory')
paraser.add_argument('--material_dir', type=str, default='poisson_fusion/lesion_library_no_edge_hd',
                     help='The path of cropped lesion patches')
paraser.add_argument('--aug_rate', type=int, default=19,
                     help='The augmentation rate of training set')
paraser.add_argument('--dens_EX', type=str, default='0_60',
                     help='The density range of pasted HE lesion, min_max')
paraser.add_argument('--dens_HE', type=str, default='0_0',
                     help='The density range of pasted MA lesion, min_max')
paraser.add_argument('--dens_MA', type=str, default='0_100',
                     help='The density range of pasted EX lesion, min_max')
paraser.add_argument('--dens_SE', type=str, default='0_0',
                     help='The density range of pasted SE lesion, min_max')
args = paraser.parse_args()
#training and testing
paraser.add_argument('--GPU_id', type=str, default='0', help='GPU id')
paraser.add_argument('--log_name', type=str,default=None, help='a log name for an exp')
paraser.add_argument('--use_pb', type=str, default='yes', help='wo(yes) or w/o(no) PBDA')
paraser.add_argument('--classes', type=int, default=5, help='number of label class')
paraser.add_argument('--target_size', type=int, default=1376, help='input size of image')
paraser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training')
paraser.add_argument('--val_batch_size', type=int, default=1, help='batch size for test')
paraser.add_argument('--val_num', type=int, default=27, help='iterations of testing in one epoch')
paraser.add_argument('--train_num', type=int, default=540, help='iterations of training in one epoch')
paraser.add_argument('--extra_aug', type=str, default='y', help='extra augmentation methods')
paraser.add_argument('--flag_multi_class', type=str, default='y', help='classes > 1?')
paraser.add_argument('--epochs', type=int, default=26, help='maximum epochs for training')
paraser.add_argument('--bg_epoch', type=int, default=0, help='beginning epoch number')
paraser.add_argument('--learning_rate', type=int, default=0.0001, help='learning rate')
paraser.add_argument('--weight_decay_rate', type=int, default=0.0001, help='weight decay rate')
paraser.add_argument('--model_name', type=str, default='unet_plusplus',
                     help='model name: unet_plusplus/unet/deeplabv3+/unet3_plus/original_unet_plusplus')
paraser.add_argument('--supervision', type=str, default='y', help='using supervision')
paraser.add_argument('--use_ps', type=str, default='y', help='using pixel-shuffling')
paraser.add_argument('--norm', type=str, default='gn', help='bn/gn')
paraser.add_argument('--loss', type=str, default='dice_CE', help='loss function')
paraser.add_argument('--train_strategy', type=str, default='step_decay', help='lr decay strategy')
paraser.add_argument('--save_result', type=str, default='y', help='save image results?')
paraser.add_argument('--task', type=str, default=None, help='task')
paraser.add_argument('--train_path', type=str, default=None, help='path of training set')
paraser.add_argument('--label_folder', type=str,
                     default='label_zoom_blend_hd_aug20_EX60_HE0_MA100_SE0',
                     help='name of label folder')
paraser.add_argument('--image_folder', type=str,
                     default='image_zoom_blend_hd_aug20_EX60_HE0_MA100_SE0',
                     help='name of image folder')
args = paraser.parse_args()
print(args)
def run(args):
    if args.preprocess:
        from preprocessing import main
        main(args)
    if args.vessel_seg:
        from vessel_seg.main import main
        main(args)
    if args.OD_seg:
        from disc_seg.disc_seg import main
        main(args)
    if args.build_lesion_lab:
        from poisson_fusion.build_material_library import main
        main(args)
    if args.build_PBDA_dataset:
        from poisson_fusion.poisson_blend import main
        main(args)
    if args.step>-1:
        if args.dataset == 'IDRiD':
            from training_and_evaluation_train_on_batch_pb import main
            main(args)
        if args.dataset == 'e_ophtha':
            from training_and_evaluation_train_on_batch_pb_for_ep import main
            args.epochs = 28
            args.target_size = 1024
            args.train_num = 140
            args.val_num = 7
            main(args)

if __name__ == '__main__':
    run(args)


