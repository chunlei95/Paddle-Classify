import time
from tqdm import tqdm
import wandb
from core.eval import evaluate
import paddle
import os


def train(model,
          train_loader,
          val_loader,
          loss_fn,
          optimizer,
          lr_scheduler,
          total_epoch,
          start_epoch,
          start_val_step,
          logger,
          checkpoint_root_path,
          checkpoint_save_num,
          best_score,
          best_score_acc,
          best_score_epoch):
    best_f1 = best_score
    best_f1_accuracy = best_score_acc
    best_model_epoch = best_score_epoch
    checkpoint_num = 0
    for epoch in range(start_epoch, total_epoch):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        train_loop = tqdm(train_loader, total=len(train_loader), ncols=110, leave=True, colour='green', unit='img')
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
        lr_scheduler.step()
        if epoch + 1 >= start_val_step:
            epoch_end = time.time()
            train_epoch_interval = epoch_end - epoch_start
            avg_loss = total_loss / len(train_loader)

            model.eval()
            val_interval, eval_avg_loss, accuracy, f1, recall, precision = evaluate(model, val_loader, loss_fn, epoch)
            wandb.log({'train_loss': avg_loss, 'eval_loss': eval_avg_loss, 'lr': optimizer.get_lr()})
            logger.info(
                "[Epoch {}]: avg train loss = {}, avg eval Loss = {}, train time cost = {}s, eval time cost = {}s, lr = {}".format(
                    epoch + 1,
                    '%.4f' % avg_loss,
                    '%.4f' % eval_avg_loss,
                    '%.4f' % train_epoch_interval,
                    '%.4f' % val_interval,
                    '%.8f' % optimizer.get_lr())
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
            checkpoint = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict(),
                epoch=epoch,
                best_score=best_f1,
                best_score_acc=best_score_acc,
                best_score_epoch=best_model_epoch
            )

            checkpoint_path = checkpoint_root_path + f'/checkpoint_{epoch + 1}.pdparams'
            paddle.save(checkpoint, checkpoint_path)

            logger.info(
                '[Epoch {}]: The evaluation metrics and current best model info are following: \n'
                '     *******************************************************************************************************\n'
                '        F1 Score  |  Accuracy  |  Recall  |  Precision  |  Best F1 Score  |  Best Accuracy  |  Best Epoch  \n'
                '     *******************************************************************************************************\n'
                '         {}   |   {}   |  {}  |   {}    |      {}        |     {}      |     {}      \n'
                '     *******************************************************************************************************'.format(
                    epoch + 1,
                    '%.4f' % f1,
                    '%.4f' % accuracy,
                    '%.4f' % recall,
                    '%.4f' % precision,
                    '%.4f' % best_f1,
                    '%.4f' % best_f1_accuracy,
                    best_model_epoch,
                )
            )

            # 移除最先保留的checkpoint
            checkpoint_num += 1
            if checkpoint_num > checkpoint_save_num:
                remove_path = 'checkpoint/checkpoint_{}.pdparams'.format(epoch + 1 - checkpoint_save_num)
                os.remove(os.path.abspath(remove_path))
                checkpoint_num -= 1
