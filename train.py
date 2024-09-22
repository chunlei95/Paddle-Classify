import argparse
import os
import random
import warnings

import numpy as np
import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
import wandb
from paddle.io import DataLoader

from core.train import train
from datasets.cropidentity import CropIdentityDataset
from datasets.pest_and_disease import PestAndDiseaseDataset
from models.RepViT import RepViT
# from models.van import VAN_B3
from models.InceptionNeXt import InceptionNeXt_T, InceptionNeXt_S, InceptionNeXt_B
from models.SHViT import SHViT_S4, SHViT_S5
from models.RMT import RMT_L6, RMT_T3, RMT_M2, RMT_S
from models.efficientformer_v2 import EfficientFormerV2_S2, EfficientFormerV2_L
from models.lggformer import LGGFormer
from models.van import VAN_B2, VAN_B0
from utils.logger import setup_logger

logger = setup_logger('Train', 'logs/train.log')

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', dest='config', type=str, help='The configuration file path')
    parser.add_argument('--lr', dest='lr', default='0.0006', type=float, help='learning rate')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--total_epoch', dest='total_epoch', default=300, type=int, help='total training epoch')
    parser.add_argument('--eval_epoch', dest='eval_epoch', default=0, type=int,
                        help='the epoch start evaluate the training model')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='random seed used for paddle and numpy global config')
    parser.add_argument('--ckpt_save_path', dest='ckpt_save_path', default='checkpoint',
                        help='the path to save checkpoints')
    parser.add_argument('--ckpt_keep_num', dest='ckpt_keep_num', type=int, default=2,
                        help='the number of checkpoint saved')
    parser.add_argument('--resume', dest='resume', default=None, type=str, help='resume checkpoint path')
    parser.add_argument('--device', dest='device', default='gpu', type=str, choices=['gpu', 'cpu'])
    parser.add_argument('--use_wandb', dest='use_wandb', default=False, action='store_true',
                        help='whether to use wandb to log metrics')
    parser.add_argument('--wandb_key', dest='wandb_key', default='4fceea5c83c7ff2e496774cc0359554fc8912e77', type=str,
                        help='the key used to login wandb')
    parser.add_argument('--pretrained_path', dest='pretrained_path', default=None, type=str)
    return parser.parse_args()


def main(args):
    paddle.device.set_device('gpu')
    # 设置全局随机种子
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        paddle.seed(args.seed)

    # data_root = 'D:/datasets/crop_identity_new'
    data_root = '/media/humrobot/Data/datasets/农作物病虫害数据集'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # train_dataset = CropIdentityDataset(data_root=data_root,
    #                                     augment_root='D:/dataset/augment_pest_and_disease_train',
    #                                     mode='train',
    #                                     transforms=train_transform)
    # val_dataset = CropIdentityDataset(data_root=data_root,
    #                                   augment_root='D:/dataset/augment_pest_and_disease_val',
    #                                   mode='val',
    #                                   transforms=val_transform)

    train_dataset = PestAndDiseaseDataset(data_root=data_root,
                                          augment_root='/media/humrobot/Data/dataset/augment_pest_and_disease_train',
                                          mode='train',
                                          transforms=train_transform)
    val_dataset = PestAndDiseaseDataset(data_root=data_root,
                                        augment_root='/media/humrobot/Data/dataset/augment_pest_and_disease_val',
                                        mode='val',
                                        transforms=val_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)

    # 对于农作物分类来说，目前最好的模型是VAN
    # model = VAN(class_num=19,
    #             drop_path_rate=0.2,
    #             drop_rate=0.,
    #             embed_dims=[64, 128, 320, 512],
    #             mlp_ratios=[8, 8, 4, 4],
    #             norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    #             depths=[3, 3, 12, 3])

    # model = RepViT(stage_channels=[64, 128, 320, 512], stage_depths=[3, 3, 21, 3])

    model = VAN_B2(pretrained=True, class_num=123, drop_path_rate=0.2, drop_rate=0.2, img_size=224)

    # model = InceptionNeXt_B(num_classes=123, in_channels=3)
    # model = SHViT_S4(num_classes=123, in_channels=3)
    # model = RMT_S(in_channels=3, num_classes=19)

    # model = LGGFormer(in_channels=3,
    #                   num_classes=19,
    #                   patch_size=3,
    #                   stage_channels=[48, 96, 192, 384],
    #                   encoder_stage_blocks=[2, 2, 4, 2],
    #                   num_heads=[2, 4, 8, 16],
    #                   trans_layers=2,
    #                   drop_path_rate=0.2,
    #                   norm_type=nn.BatchNorm2D,
    #                   act_type=nn.ReLU6)

    # model = EfficientFormerV2_L(in_channels=3, num_classes=123)

    # model = VAN_B3(class_num=19, drop_path_rate=0.2, drop_rate=0.2)
    # model = NextViT_base_224(class_num=19, attn_drop=0.2)
    # model = ConvNeXt_base_224(class_num=19, drop_path_rate=0.2)

    if args.pretrained_path:
        model_params = paddle.load(args.pretrained_path)
        model.set_state_dict(model_params)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # lr_scheduler_post = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.total_epoch - 5)
    # lr_scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=lr_scheduler_post, warmup_steps=5, start_lr=1.0e-6,
    #                                                 end_lr=args.lr)

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.total_epoch)

    # optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=lr_scheduler)

    if not os.path.exists(args.ckpt_save_path):
        os.makedirs(args.ckpt_save_path)

    start_epoch = 0
    best_score = 0.
    best_score_acc = 0.
    best_score_epoch = 0
    if args.resume is not None:
        ckpt_params = paddle.load(args.resume)
        model.set_state_dict(ckpt_params['model'])
        optimizer.set_state_dict(ckpt_params['optimizer'])
        lr_scheduler.set_state_dict(ckpt_params['lr_scheduler'])
        start_epoch = ckpt_params['epoch'] + 1
        best_score = ckpt_params['best_score']
        best_score_acc = ckpt_params['best_score_acc']
        best_score_epoch = ckpt_params['best_score_epoch']

    if args.use_wandb:
        if args.wandb_key is not None:
            wandb.login(key=args.wandb_key)
        else:
            wandb.login(anonymous='allow')
        wandb.init(project="crop-identity",
                   config={
                       "batch_size": args.batch_size,
                       "epochs": args.total_epoch,
                       "lr": args.lr,
                       "optimizer": "AdamW",
                       "loss": "CrossEntropyLoss"
                   })

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          lr_scheduler=lr_scheduler,
          logger=logger,
          use_wandb=args.use_wandb,
          total_epoch=args.total_epoch,
          start_epoch=start_epoch,
          start_val_step=args.eval_epoch,
          checkpoint_root_path=args.ckpt_save_path,
          checkpoint_save_num=args.ckpt_keep_num,
          best_score=best_score,
          best_score_acc=best_score_acc,
          best_score_epoch=best_score_epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)
