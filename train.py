import os
import random
import time
import warnings

import numpy as np
import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
import wandb
from paddle.io import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from datasets.cropidentity import CropIdentityDataset
from models.nextvit import NextViT_base_224
from utils.logger import setup_logger

logger = setup_logger('train', 'logs/train.log')

warnings.filterwarnings('ignore')


def main():
    seed = 42
    batch_size = 64
    lr = 0.0006
    epoch = 200
    checkpoint_save_num = 3
    paddle.device.set_device('gpu')
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    data_root = 'crop_identity_new'
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
    train_dataset = CropIdentityDataset(data_root=data_root, augment_root='/kaggle/working/augment_crop_identity',
                                        mode='train', transforms=train_transform)
    val_dataset = CropIdentityDataset(data_root=data_root, augment_root='', mode='val', transforms=val_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, batch_size=batch_size)
    # model = VAN(class_num=19, drop_path_rate=0.1, drop_rate=0.1)
    model = NextViT_base_224(class_num=19, attn_drop=0.2)
    loss_fn = nn.CrossEntropyLoss()

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epoch)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=lr_scheduler)
    best_f1 = 0.
    best_f1_accuracy = 0.
    best_model_epoch = 0
    checkpoint_num = 0
    checkpoint_root_path = './checkpoint'
    if not os.path.exists(checkpoint_root_path):
        os.makedirs(checkpoint_root_path)
    wandb_key = '4fceea5c83c7ff2e496774cc0359554fc8912e77'
    wandb.login(key=wandb_key)
    wandb.init(project="crop-identity",
               config={
                   "batch_size": batch_size,
                   "epochs": epoch,
                   "lr": lr,
                   "optimizer": "AdamW",
                   "loss": "CrossEntropyLoss"
               })
    for epoch in range(epoch):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        train_loop = tqdm(train_loader, total=len(train_loader), ncols=110, leave=False, colour='blue')
        for epoch_iter, batch in enumerate(train_loop):
            iter_start = time.time()
            inputs, labels = batch
            labels = labels.squeeze(-1)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            iter_end = time.time()
            iter_interval = iter_end - iter_start
            train_loop.set_description('[Epoch {}]({}/{})'.format(epoch + 1, epoch_iter + 1, len(train_loader)))
            train_loop.set_postfix_str('iter train loss = {}, batch cost= {} s'.format(
                '%.4f' % loss.item(),
                '%.4f' % iter_interval)
            )
        train_loop.clear()
        lr_scheduler.step()
        end_epoch = time.time()
        train_epoch_interval = end_epoch - epoch_start
        avg_loss = total_loss / len(train_loader)
        model.eval()
        with paddle.no_grad():
            val_start = time.time()
            eval_total_loss = 0.0
            predict_list = None
            label_list = None
            val_loop = tqdm(val_loader, total=len(val_loader), ncols=150, leave=False, colour='blue')
            for val, eval_batch in enumerate(val_loop):
                val_iter_start = time.time()
                eval_inputs, eval_labels = eval_batch
                eval_labels = eval_labels.squeeze()
                eval_outputs = model(eval_inputs)
                val_iter_end = time.time()
                val_iter_interval = val_iter_end - val_iter_start
                eval_loss = loss_fn(eval_outputs, eval_labels)
                eval_total_loss += eval_loss.item()
                eval_outputs = paddle.argmax(eval_outputs, axis=1)
                if predict_list is None:
                    predict_list = eval_outputs
                    label_list = eval_labels
                else:
                    predict_list = paddle.concat((predict_list, eval_outputs), axis=0)
                    label_list = paddle.concat((label_list, eval_labels), axis=0)
                val_loop.set_description('[Valid {}]({}/{})'.format(epoch + 1, val + 1, len(val_loader)))
                val_loop.set_postfix_str(
                    'iter valid loss = {}, batch cost = {} s, infer time = {} ms/image'.format(
                        '%.4f' % eval_loss.item(),
                        '%.4f' % val_iter_interval,
                        '%.4f' % (val_iter_interval / eval_inputs.shape[0] * 1000))
                )
            val_loop.clear()
        val_end = time.time()

        val_interval = val_end - val_start

        eval_avg_loss = eval_total_loss / len(val_loader)

        wandb.log({'train_loss': avg_loss, 'eval_loss': eval_avg_loss, 'lr': optimizer.get_lr()})
        logger.info(
            "[Epoch {}]: avg train loss = {}, avg eval Loss = {}, train time cost = {}s, eval time cost = {}s, lr update to {}".format(
                epoch + 1,
                '%.4f' % avg_loss,
                '%.4f' % eval_avg_loss,
                '%.4f' % train_epoch_interval,
                '%.4f' % val_interval,
                '%.8f' % optimizer.get_lr())
        )

        metrics_dict = classification_report(label_list.cpu(), predict_list.cpu(), output_dict=True, zero_division=0)

        accuracy = metrics_dict.get('accuracy')
        metrics_dict = metrics_dict.get('macro avg')
        recall, precision, f1 = metrics_dict.get('recall'), metrics_dict.get('precision'), metrics_dict.get('f1-score')
        wandb.log({'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})

        logger.info(
            '[Epoch {}]: The evaluation metrics are following: \n'
            '     ****************************************************\n'
            '     |  Accuracy  |   Recall  |  Precision  |  F1 score |\n'
            '     ****************************************************\n'
            '     |   {}   |   {}  |   {}    |   {}  |\n'
            '     ****************************************************'.format(
                epoch + 1, '%.4f' % accuracy,
                '%.4f' % recall, '%.4f' % precision,
                '%.4f' % f1))

        checkpoint = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler.state_dict(),
            epoch=epoch,
            loss=avg_loss,
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            f1=f1
        )

        # 更新最优模型
        best_model_path = checkpoint_root_path + '/best_model.pdparams'
        if f1 > best_f1:
            best_f1 = f1
            best_f1_accuracy = accuracy
            best_model_epoch = epoch + 1
            paddle.save(model.state_dict(), best_model_path)
            logger.info('[Epoch {}]: save best model to {}'.format(epoch + 1, os.path.abspath(best_model_path)))

        # 保存当前断点
        checkpoint_path = checkpoint_root_path + f'/checkpoint_{epoch + 1}.pdparams'
        paddle.save(checkpoint, checkpoint_path)
        logger.info(
            '[Epoch {}]: Model saved to {}, current best model infos are following: \n'
            '     ****************************************************\n'
            '     |  Best Epoch  |  Best F1 Score  |  Best Accuracy  |\n'
            '     ****************************************************\n'
            '     |     {}        |      {}     |      {}     |\n'
            '     ****************************************************'.format(
                epoch + 1,
                os.path.abspath(checkpoint_path),
                best_model_epoch,
                '%.4f' % best_f1,
                '%.4f' % best_f1_accuracy
            )
        )

        checkpoint_num += 1
        if checkpoint_num > checkpoint_save_num:
            remove_path = 'checkpoint/checkpoint_{}.pdparams'.format(epoch + 1 - checkpoint_save_num)
            os.remove(os.path.abspath(remove_path))
            checkpoint_num -= 1
            logger.info(f"[Epoch {epoch + 1}]: Model removed from {os.path.abspath(remove_path)}")


if __name__ == "__main__":
    main()
