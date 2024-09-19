import paddle
import time
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report


def evaluate(model,
             val_loader,
             loss_fn,
             epoch,
             use_wandb):
    with paddle.no_grad():
        val_start = time.time()
        eval_total_loss = 0.0
        predict_list = None
        label_list = None
        val_loop = tqdm(val_loader, total=len(val_loader), ncols=150, leave=False, colour='green', unit='img')
        for val, eval_batch in enumerate(val_loop):
            val_iter_start = time.time()
            eval_inputs, eval_labels = eval_batch
            # eval_labels = eval_labels.squeeze()
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

    metrics_dict = classification_report(label_list.cpu(), predict_list.cpu(), output_dict=True, zero_division=0)
    accuracy = metrics_dict.get('accuracy')
    metrics_dict = metrics_dict.get('macro avg')
    recall, precision, f1 = metrics_dict.get('recall'), metrics_dict.get('precision'), metrics_dict.get('f1-score')
    if use_wandb:
        wandb.log({'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1})

    return val_interval, eval_avg_loss, accuracy, f1, recall, precision