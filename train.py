import os
import sys
import numpy as np
import argparse
import time
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.fs_loader import FewShotLoader, sampler
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.fsod_logger import FSODLogger

from utils import *
from model.utils.metaclm import SupConLoss
# import faster-rcnn
    
if __name__ == '__main__':
    criterion = SupConLoss(temperature=0.07)
    #args = parse_args()
    #这里直接写死就行 或者写个变量 然后自定义就行
    dataset = "voc1"
    flip = True #是否使用翻转的数据
    net = "hanmcl"
    lr = 0.001
    lr_decay_step = 12
    bs = 4
    epochs = 10
    disp_interval = 20
    save_dir = "models/hANMCL"
    way = 2
    shot = 10
    #
    custom_args = f"--dataset {dataset} {'--flip' if flip else ''} --net {net} --lr {lr} --lr_decay_step {lr_decay_step} --bs {bs} --epochs {epochs} --disp_interval {disp_interval} --save_dir {save_dir} --way {way} --shot {shot}"
    # custom_args = "--dataset voc1 --flip --net hanmcl --lr 0.001 --lr_decay_step 12 --bs 4 --epochs 10 --disp_interval 20 --save_dir models/hANMCL --way 2 --shot 10"

    # for i in range(0,len(custom_args),2):
    #     args = parse_args(str(custom_args[i]+" "+custom_args[i+1]))
    args = parse_args(custom_args)#改了个函数 可以完成args的手动输入

    print(args.shot)

    cfg_from_file(args.cfg_file)
    cfg_from_list(args.set_cfgs)

    # make results determinable
    random_seed = 1996
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) #这个好像没关系
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.CUDA = True

    # prepare output dir
    output_dir = os.path.join(args.save_dir, "train/checkpoints") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare dataloader
    cfg.TRAIN.USE_FLIPPED = args.use_flip #是否翻转
    cfg.USE_GPU_NMS = True
    #这个roi数据是怎么生成的呢？
    #能不能把roi这个东西直接淘汰掉 自己写一个 dataloader 反正就是提供  和别的fewshotloader尽量靠拢就行 因为他没给json
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name) #ROI数据用来训练


    dataset = FewShotLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                            imdb.num_classes, training=True, num_way=args.way, num_shot=args.shot)
    
    
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sampler_batch = sampler(train_size, args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holders  把这些参数 全都获取 并放到gpu上去
    holders = prepare_var(support=True)
    im_data = holders[0]
    im_info = holders[1]
    num_boxes = holders[2]
    gt_boxes = holders[3]
    support_ims = holders[4]

    # initilize the network
    pre_weight = False if args.resume else True
    classes = ['fg', 'bg']
    model = get_model(args.net, pretrained=pre_weight, way=args.way, shot=args.shot, classes=classes)
    #有gpu则放到cuda上，没有则注释掉
    #model.cuda()

    # optimizer
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # load checkpoints 
    if args.resume:
        load_dir = os.path.join(args.load_dir, "train/checkpoints")
        load_name = os.path.join(load_dir, f'model_{args.checkepoch}_{args.checkpoint}.pth')
        checkpoint = torch.load(load_name)
        args.start_epoch = 0
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print(f'loaded checkpoint: {load_name}')

    if args.mGPUs:
        model = nn.DataParallel(model)

    # initialize logger
    if not args.dlog:
        logger_save_dir = os.path.join(args.save_dir, "train")
        tb_logger = FSODLogger(logger_save_dir)

    # training
    iters_per_epoch = int(train_size / args.batch_size)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        model.train()
        loss_temp = 0
        start_time = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
                support_ims.resize_(data[4].size()).copy_(data[4])

            model.zero_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, correlation_feat2 = model(im_data, im_info, gt_boxes, num_boxes, support_ims)
   
            cont = criterion(correlation_feat2)*0.07

            loss = cont.mean() + rpn_loss_cls.mean() + rpn_loss_box.mean() \
                + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end_time = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                        % (epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end_time-start_time))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, metric %.4f" \
                            % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, cont))

                info = {
                'loss': loss_temp,
                'loss_rpn_cls': loss_rpn_cls,
                'loss_rpn_box': loss_rpn_box,
                'loss_rcnn_cls': loss_rcnn_cls,
                'loss_rcnn_box': loss_rcnn_box
                }
                loss_temp = 0
                start_time = time.time()
        if not args.dlog:
            tb_logger.write(epoch, info, save_im=args.imlog)

        save_name = os.path.join(output_dir, 'model_{}_{}.pth'.format(epoch, step))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
        }, save_name)
        print('save model: {}'.format(save_name))



