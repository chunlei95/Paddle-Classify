import cv2  # 导入OpenCV库
import numpy as np
def CannyThreshold(lowThreshold):
    # 使用3x3的高斯滤波对灰度图进行模糊处理
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    # 使用Canny边缘检测，lowThreshold为下阈值，ratio为高低阈值比例，kernel_size为Sobel算子的大小
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    # detected_edges = cv2.Canny(detected_edges, 20, 60, apertureSize=kernel_size)
    # 使用位运算将边缘检测结果叠加到原图上
    # dst = cv2.bitwise_and(img, img, mask=detected_edges)
    dst = cv2.bitwise_and(gray, gray, mask=detected_edges)
    cv2.imwrite('result' + str(lowThreshold) + '.png', dst)

    # #开放区域闭合
    # # 定义结构元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # # 膨胀操作
    # # dilated = cv2.dilate(dst, kernel, iterations=1)
    # # 闭操作
    # closed = cv2.morphologyEx(dst,cv2.MORPH_CLOSE, kernel)
    # # 显示结果
    # cv2.imwrite('result_close' + str(lowThreshold) + '.png', closed)

    #小面积过滤
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓，删除小面积区域
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            cv2.drawContours(dst, [contour], 0, (0, 0, 0), -1)

    # cv2.imwrite('result_area' + str(lowThreshold) + '.png', labels.astype(np.uint8)*255)
    # cv2.imwrite('result_area' + str(lowThreshold) + '.png', labels.astype(np.uint8)*255)
    # 修改前
    # src_y = round(dst_y * src_height / tar_height)
    # src_x = round(dst_x * src_width / tar_width)
    # 修改后
    # src_y = round(dst_y * src_height / tar_height + 1 / 2 * (src_height / tar_height - 1))
    # src_x = round(dst_x * src_width / tar_width + 1 / 2 * (src_height / tar_height - 1))

    # 返回处理后的图像
    # result = cv2.bitwise_and(gray, gray, mask=labels.astype(np.uint8)*255)
    result = cv2.bitwise_and(gray, gray, mask=dst)

    # 在窗口中显示Canny边缘检测的结果
    cv2.imshow('canny demo', result)
    cv2.imwrite('result_area' + str(lowThreshold) + '.png', result)
    # cv2.imwrite('result_area' + str(lowThreshold) + '.png', labels.astype(np.uint8)*255)

    # cv2.imshow('canny demo', detected_edges)





# 初始化变量
lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

# 读取图像并将其转换为灰度图像
img = cv2.imread('test_csx.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个窗口
cv2.namedWindow('canny demo')

'''
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字 
第三个参数，是这个trackbar的默认值,也是调节的对象 
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名 
'''
# 创建一个trackbar，用于调节Canny边缘检测的阈值
cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
# 初始时调用一次CannyThreshold函数，显示初始阈值下的Canny边缘检测结果
CannyThreshold(0)
# 等待用户按键操作，按下ESC键则关闭窗口
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


# def nearest_demo(src, multiple_y, multiple_x):
#     src_y, src_x, src_c = src.shape
#     tar_x, tar_y, tar_c = src_x * multiple_x, src_y * multiple_y, src_c
#     # 生成一个黑色的目标图像
#     tar_img = np.zeros((tar_y, tar_x, tar_c), dtype=np.uint8)
#     print(tar_img.shape)
#     # 渲染像素点的值
#     # 注意 y 是高度，x 是宽度
#     for y in range(tar_y - 1):
#         for x in range(tar_x - 1):
#             # 计算新坐标 (x,y) 坐标在源图中是哪个值
#
#             src_y = round(y * src_y / tar_y)
#             src_x = round(x * src_x / tar_x)
#
#             tar_img[y, x] = src[src_y, src_x]
#
#     return tar_img



