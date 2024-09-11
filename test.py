import time
import warnings

import paddle
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

from datasets.cropidentity import CropIdentityDataset
from models.van import VAN_B3
from utils.logger import setup_logger

warnings.filterwarnings('ignore')

logger = setup_logger()


def main():
    batch_size = 64
    checkpoint_path = 'D:/PycharmProjects/van_b3_crop_identity.pdparams'
    paddle.device.set_device('gpu')
    data_root = 'D:/datasets/crop_identity_new'
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = CropIdentityDataset(data_root=data_root, augment_root='', mode='test', transforms=test_transform)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, batch_size=batch_size)
    # model = VAN(class_num=19,
    #             drop_path_rate=0.2,
    #             drop_rate=0.2,
    #             embed_dims=[64, 128, 320, 512],
    #             mlp_ratios=[8, 8, 4, 4],
    #             norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
    #             depths=[3, 3, 12, 3])
    model = VAN_B3(class_num=19, img_size=256)
    params = paddle.load(checkpoint_path)
    model.set_state_dict(params)
    model.eval()
    with paddle.no_grad():
        val_start = time.time()
        predict_list = None
        label_list = None
        val_loop = tqdm(test_loader, total=len(test_loader), ncols=150, leave=False, colour='green', unit='img')
        for val, eval_batch in enumerate(val_loop):
            val_iter_start = time.time()
            eval_inputs, eval_labels = eval_batch
            # eval_labels = eval_labels.squeeze()
            eval_outputs = model(eval_inputs)
            val_iter_end = time.time()
            val_iter_interval = val_iter_end - val_iter_start
            eval_outputs = paddle.argmax(eval_outputs, axis=1)
            if predict_list is None:
                predict_list = eval_outputs
                label_list = eval_labels
            else:
                predict_list = paddle.concat((predict_list, eval_outputs), axis=0)
                label_list = paddle.concat((label_list, eval_labels), axis=0)
            val_loop.set_description('[Test]({}/{})'.format(val + 1, len(test_loader)))
            val_loop.set_postfix_str(
                'batch cost = {} s, infer time = {} ms/image'.format(
                    '%.4f' % val_iter_interval,
                    '%.4f' % (val_iter_interval / eval_inputs.shape[0] * 1000))
            )
        val_loop.clear()
    val_end = time.time()
    metrics_dict = classification_report(label_list.cpu(), predict_list.cpu(), output_dict=True, zero_division=0)

    accuracy = metrics_dict.get('accuracy')
    metrics_dict = metrics_dict.get('macro avg')
    recall, precision, f1 = metrics_dict.get('recall'), metrics_dict.get('precision'), metrics_dict.get('f1-score')
    logger.info('[Test]: Total {} images, total test time is {} s'.format(len(predict_list), val_end - val_start, ))
    logger.info(
        '[Test]: The evaluation metrics and current best model info are following: \n'
        '     ****************************************************\n'
        '     |  F1 Score  |  Accuracy  |  Recall  |  Precision  |\n'
        '     ****************************************************\n'
        '     |   {}   |   {}   |  {}  |   {}    |\n'
        '     ****************************************************'.format(
            '%.4f' % f1,
            '%.4f' % accuracy,
            '%.4f' % recall,
            '%.4f' % precision,
        )
    )


if __name__ == '__main__':
    main()
