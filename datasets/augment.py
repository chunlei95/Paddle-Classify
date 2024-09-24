import os.path
import random
import uuid

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


# gamma变换
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


# 随机gamma变换
def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


# 旋转图像
def rotate(xb, angle, img_w, img_h):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    return xb


# 图像模糊
def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


# 添加噪声
def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def randomColor(image, saturation=0, brightness=0, contrast=0, sharpness=0):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if random.random() < saturation:
        random_factor = np.random.randint(0, 11) / 10.  # 随机因子
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(10, 11) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(10, 11) / 10.  # 随机因1子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(0, 11) / 10.  # 随机因子
        image = ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def randomRotation(xb, img_w, img_h, mode=Image.BICUBIC):
    """
    对图像进行随机任意角度(0~360度)旋转
    :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    """
    angle = np.random.randint(1, 360)
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    return xb


def data_augment(xb, img_w, img_h, rate=0.15):
    # 随机旋转
    if np.random.random() < rate:
        xb = rotate(xb, 90, img_w, img_h)
    if np.random.random() < rate:
        xb = rotate(xb, 180, img_w, img_h)
    if np.random.random() < rate:
        xb = rotate(xb, 270, img_w, img_h)
    # 上下翻转
    if np.random.random() < rate:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
    # gamma变换
    if np.random.random() < rate:
        xb = random_gamma_transform(xb, 1.0)
    # 模糊
    if np.random.random() < rate:
        xb = blur(xb)
    # 噪声
    if np.random.random() < rate:
        xb = add_noise(xb)
    # 调整饱和度
    if np.random.random() < rate:
        xb = randomColor(xb, saturation=1, brightness=0, contrast=0, sharpness=0)
    # 调整亮度
    if np.random.random() < rate:
        xb = randomColor(xb, saturation=0, brightness=1, contrast=0, sharpness=0)
    # 调整对比度
    if np.random.random() < rate:
        xb = randomColor(xb, saturation=0, brightness=0, contrast=1, sharpness=0)
    # 调整锐度
    if np.random.random() < rate:
        xb = randomColor(xb, saturation=0, brightness=0, contrast=0, sharpness=1)
    return xb


def create_train_dataset(save_path, max_num_per_class, count_dict: dict, img_dict: dict):
    # max_num_class = max(count_dict.values())
    # min_num_class = min(count_dict.values())

    # max_num_per_class = (sum(count_dict.values()) - max_num_class - min_num_class) // (len(count_dict) - 2)

    # max_num_per_class = sum(count_dict.values()) // len(count_dict)

    # max_num_per_class = max(count_dict.values())
    for k, v in count_dict.items():
        k_items = [k_p for k_p, v_p in img_dict.items() if v_p == k]
        for j in range(len(k_items)):
            if j > max_num_per_class:
                break
            ori_img_path = k_items[j]
            _, file_name = os.path.split(ori_img_path)
            new_path = save_path + '/' + k
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            new_path = new_path + '/' + file_name
            try:
                ori_img = Image.open(ori_img_path)
                # img_arr = cv2.imdecode(np.fromfile(ori_img_path), cv2.IMREAD_COLOR)
                # ori_img = ori_img.convert('RGB')
                img_arr = np.array(ori_img)
                if img_arr.shape[-1] == 4:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                else:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                cv2.imencode('.jpg', img_arr)[1].tofile(new_path)
            except:
                print(ori_img_path)
                pass
            # cv2.imwrite(new_path, img_arr)
        if v < max_num_per_class:
            diff_num = max_num_per_class - v
            aug_loop = tqdm(range(diff_num), total=diff_num, colour='green', leave=True, unit='img')
            for i in aug_loop:
                idx = np.random.randint(0, v)
                ori_img_path = k_items[idx]
                _, file_name = os.path.split(ori_img_path)
                new_path = save_path + '/' + k
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                new_path = new_path + '/' + file_name
                try:
                    ori_img = Image.open(ori_img_path)
                    # ori_img = ori_img.convert('RGB')
                    img_arr = np.array(ori_img)
                    if img_arr.shape[-1] == 4:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                    else:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                    img_h, img_w, _ = img_arr.shape
                    new_img = data_augment(img_arr.copy(), img_w, img_h)
                    name, ext = os.path.splitext(new_path)
                    name = name + '_' + uuid.uuid4().hex
                    save_img_path = name + ext
                    cv2.imencode('.jpg', new_img)[1].tofile(save_img_path)
                    # cv2.imwrite(save_img_path, new_img)
                    aug_loop.set_description('Augment images of class {}'.format(k))
                except:
                    print(ori_img_path)
                    pass
