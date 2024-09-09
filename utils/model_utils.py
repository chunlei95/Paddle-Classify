import time

import paddle


def count_syncbn(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += int(2 * nelements)


def calculate_flops_and_params(images, model, precision='fp32', amp_level='01'):
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if local_rank == 0 and not (precision == 'fp16' and amp_level == 'O2'):
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w],
            custom_ops={paddle.nn.SyncBatchNorm: count_syncbn})


def calculate_inference_time(images, model, pretrain_params):
    model_params = paddle.load(pretrain_params)
    model.set_stage_dict(model_params)
    start = time.time()
    predict = model(images)
    end = time.time()
    return start - end

# def label_map(label_path, label_maps, save_path):
#     # label = imageio.v2.imread(label_path)
#     img = Image.open(label_path).convert('RGB')
#     im_arr = np.asarray(img)
#     label = im_arr.copy()
#     for key in label_maps.keys():
#         label[label == key] = label_maps[key]
#     label = label.astype(np.float32)
#     img = Image.fromarray(label)
#
#     img.save(save_path)
#
#     # imageio.imwrite(save_path, label * 255, format='RGB')
#
#
# if __name__ == '__main__':
# label_path = "D:/dataset/ACDC/patient001_frame01_slice5.png"
# label_maps = {
#     1: 100,
#     2: 80,
#     3: 128
# }
# save_path = 'D:/dataset/ACDC/1.jpg'
# label_map(label_path, label_maps, save_path)
