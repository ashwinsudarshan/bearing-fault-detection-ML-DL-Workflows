# 生成数据集图片和对应的标签
# 图片名称 对应  标签名称
# 对应版本 VGG16
# 输入：224x224的RGB 彩色图像
import pywt
import os

import matplotlib.pyplot as plt
from joblib import dump, load
import numpy as np
from PIL import Image


# 生成时频图片
def makeTimeFrequencyImage(data, img_path, img_size):
    # 数据长度
    sampling_length = len(data)
    # 连续小波变换参数
    # 采样频率
    sampling_period = 1.0 / 12000
    # 尺度长度
    totalscal = 128
    # 小波基函数
    wavename = 'cmor100-1'
    # 小波函数中心频率
    fc = pywt.central_frequency(wavename)
    # 常数c
    cparam = 2 * fc * totalscal
    # 小波尺度序列
    scales = cparam / np.arange(totalscal, 0, -1)
    # 进行连续小波变换
    coefficients, frequencies = pywt.cwt(data, scales, wavename, sampling_period)
    # 系数矩阵绝对值
    amp = abs(coefficients)
    # 生成图片
    # 根据采样频率 sampling_period 生成时间轴 t
    t = np.linspace(0, sampling_period, sampling_length, endpoint=False)
    plt.contourf(t, frequencies, amp, cmap='jet')
    plt.axis('off')  # 设置图像坐标轴不可见
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 调整子图的位置和间距，将子图充满整个图像
    plt.margins(0, 0)  # 设置图像的边距为0，即没有额外的空白边框。
    plt.gcf().set_size_inches(img_size / 100, img_size / 100)  # 设置图像的大小，单位为英寸
    plt.savefig(img_path, dpi=100)
    plt.clf()  # 避免内存溢出
    plt.close()  # 释放内存


# 生成图片数据集
def GenerateImageDataset(path_list, data_set):
    # 遍历数据集和对应目录
    for index in range(len(data_set)):
        dataset = data_set[index]
        # 更改列名
        dataset.columns = ['denormal', 'de7inner', 'de7ball', 'de7outer', 'de14inner',
                           'de14ball', 'de14outer', 'de21inner', 'de21ball', 'de21outer']
        path = path_list[index]
        # 遍历数据集每一列
        for column_name, column_data in dataset.items():
            # 行名计数器  对应每个数据名称
            count = 1
            # 遍历数据列每一行 生成一张对应的时频图片
            for row_data in dataset[column_name]:
                # 图片名称和路径
                img_path = path + column_name + '_' + str(count)
                # 生成时频图片
                makeTimeFrequencyImage(row_data, img_path, 224)
                # 图片名称+1
                count += 1





# 4通道的图片转换成3通道的
def ConvertRGB(path_list):
    for path in path_list:
        all_images = os.listdir(path)
        for image in all_images:
            image_path = os.path.join(path, image)
            img = Image.open(image_path)  # 打开图片
            # print(img.format, img.size, img.mode)#打印出原图格式
            img = img.convert("RGB")  # 4通道转化为rgb三通道
            img.save(path + image)

if __name__ == '__main__':

    # 导入数据
    train_set = load('train_set')
    val_set = load('val_set')
    test_set = load('test_set')

    data_set = [train_set, val_set, test_set]

    # 数据集路径
    train_path = 'CWTImages/train/'
    val_path = 'CWTImages/val/'
    test_path = 'CWTImages/test/'
    path_list = [train_path, val_path, test_path]

    # 第一步，生成图片
    GenerateImageDataset(path_list, data_set)
    # 第二步，进行转换
    ConvertRGB(path_list)

    # 可能会出现 警告，只要没报错停止， 就等程序运行结束 就行！！！
    # 程序运行耗时约 3分钟， 慢慢等待（时长跟自己电脑设备 也有关系）