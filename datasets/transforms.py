import cv2
import numpy as np

from paddle.vision import BaseTransform
from PIL import Image


class GaussBlur(BaseTransform):
    """Resize the input Image to the given size.

    Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int|str, optional): Interpolation method. Default: 'bilinear'.
            when use pil backend, support method are as following:
            - "nearest": Image.NEAREST,
            - "bilinear": Image.BILINEAR,
            - "bicubic": Image.BICUBIC,
            - "box": Image.BOX,
            - "lanczos": Image.LANCZOS,
            - "hamming": Image.HAMMING
            when use cv2 backend, support method are as following:
            - "nearest": cv2.INTER_NEAREST,
            - "bilinear": cv2.INTER_LINEAR,
            - "area": cv2.INTER_AREA,
            - "bicubic": cv2.INTER_CUBIC,
            - "lanczos": cv2.INTER_LANCZOS4
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Shape:
        - img(PIL.Image|np.ndarray|Paddle.Tensor): The input image with shape (H x W x C).
        - output(PIL.Image|np.ndarray|Paddle.Tensor): A resized image.

    Returns:
        A callable object of Resize.
    """

    def __init__(self, ksize=(5, 5), keys=None):
        super().__init__(keys)
        self.ksize = ksize

    def _apply_image(self, img):
        img = np.array(img)
        img = cv2.GaussianBlur(img, self.ksize, 0)
        return Image.fromarray(img)
