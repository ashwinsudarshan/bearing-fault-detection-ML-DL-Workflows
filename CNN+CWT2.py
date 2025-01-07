import torch
import torchvision
from joblib import dump, load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import PIL
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# 导入模型
from vggmodel import CNNModel


class MakeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    # torchvision.transforms.Resize(size = (224,224)),       #尺寸规范
                    torchvision.transforms.ToTensor(),  # 转化为tensor
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)  # 列出所有图片命名

    def __getitem__(self, idx: int):

        img_path = self.path_list[idx]
        # if self.train_flag is True:
        # 例如 img_path 值 cat.10844.jpg -> label = 0
        # de7ball_1.png
        if img_path.split('_')[0] == 'denormal':
            label = 0
        elif img_path.split('_')[0] == 'de7inner':
            label = 1
        elif img_path.split('_')[0] == 'de7ball':
            label = 2
        elif img_path.split('_')[0] == 'de7outer':
            label = 3
        elif img_path.split('_')[0] == 'de14inner':
            label = 4
        elif img_path.split('_')[0] == 'de14ball':
            label = 5
        elif img_path.split('_')[0] == 'de14outer':
            label = 6
        elif img_path.split('_')[0] == 'de21inner':
            label = 7
        elif img_path.split('_')[0] == 'de21ball':
            label = 8
        else:
            label = 9

        label = torch.tensor(label, dtype=torch.int64)  # 把标签转换成int64
        img_path = os.path.join(self.data_path, img_path)  # 合成图片路径
        img = PIL.Image.open(img_path)  # 读取图片
        img = self.transform(img)  # 把图片转换成tensor
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)  # 返回图片数量


# Xavier方法 初始化网络参数，最开始没有初始化一直训练不起来。
def init_normal(m):
    if type(m) == torch.nn.Linear:
        # Xavier初始化
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if type(m) == torch.nn.Conv2d:
        # Xavier初始化
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def moedel_train(train_loader, val_loader, model, parameter):
    '''
        参数
        train_loader：训练集
        val_loader：验证集
        model：模型
        parameter： 参数
        返回
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 参数初始化
    model.apply(init_normal)

    # 参数
    batch_size = parameter['batch_size']
    epochs = parameter['epochs']
    # 定义损失函数和优化函数
    loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=parameter['learn_rate'])  # 优化器
    # 初始化
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size
    # 最高准确率  最佳模型 最后的模型
    best_accuracy = 0.0
    best_model = model
    last_model = model

    train_loss = []  # 记录在训练集上每个epoch的loss的变化情况
    train_acc = []  # 记录在训练集上每个epoch的准确率的变化情况
    validate_acc = []  # 记录在验证集上每个epoch的loss的变化情况
    validate_loss = []  # 记录在验证集上每个epoch的准确率的变化情况
    print('*'*20, '开始训练', '*'*20)
    # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()

        print(f"epoch--:{epoch}")
        loss_epoch = 0.  # 保存当前epoch的loss和
        correct_epoch = 0  # 保存当前epoch的正确个数和
        for j, (seq, labels) in enumerate(train_loader):

            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)  # 压缩维度：得到输出，并将维度为1的去除   torch.Size([256, 10])

            # 计算当前batch预测正确个数
            correct_epoch += torch.sum(y_pred.argmax(dim=1).view(-1) == labels.view(-1)).item()
            # 损失计算
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

        # 计算准确率
        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            loss_validate = 0.
            correct_validate = 0
            for j, (data, label) in enumerate(val_loader):
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 计算当前batch预测正确个数
                correct_validate += torch.sum(pre.argmax(dim = 1).view(-1) == label.view(-1)).item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()

            val_accuracy = correct_validate / val_size
            print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            # 如果当前模型的准确率优于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model  # 更新最佳模型的参数

    last_model = model
    print('*' * 20, '训练结束', '*' * 20)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    print(f'best_accuracy: {best_accuracy}')
    # 保存训练图
    plt.plot(range(epochs), train_loss, color='b', label='train_loss')
    plt.plot(range(epochs), train_acc, color='g', label='train_acc')
    plt.plot(range(epochs), validate_loss, color='y', label='validate_loss')
    plt.plot(range(epochs), validate_acc, color='r', label='validate_acc')
    plt.legend()
    # plt.show()  # 显示 lable
    plt.savefig('VGG-Train', dpi=100)
    # 保存 训练过程数据
    dump(train_loss, 'train_loss')
    dump(train_acc, 'train_acc')
    dump(validate_loss, 'validate_loss')
    dump(validate_acc, 'validate_acc')
    return  last_model, best_model


if __name__ == '__main__':
    # 参数与配置
    torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性

    # 制作数据集
    # 数据集路径
    train_path = 'CWTImages/train/'
    val_path = 'CWTImages/val/'
    test_path = 'CWTImages/test/'
    # 划分图片训练集、验证集、测试集
    train_dataset = MakeDataset(train_path)
    val_dataset = MakeDataset(val_path, train=False)
    test_dataset = MakeDataset(test_path, train=False)

    # 训练参数设置
    batch_size = 32
    epochs = 3 # 自己训练的时候适当大一些（50-100）
    learn_rate = 1e-5
    num_classes = 10 # 十分类


    # VGG 参数
    conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # vgg16
    input_channels = 3

    # 制作参数字典
    parameter = {
        'batch_size': batch_size,
        'output_size': num_classes,
        'epochs': epochs,
        'learn_rate': learn_rate
    }

    # 定义模型
    # 创建 VGG16 模型
    model = CNNModel(conv_arch, num_classes)  # 定义网络

    # 加载数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    # 保存测试数据
    dump(test_loader, 'test_loader')


    last_model, best_model = moedel_train(train_loader, val_loader, model, parameter)
    # 保存最后的参数
    # torch.save(last_model, 'vgg_final_model.pt')
    # 保存最好的参数
    torch.save(best_model, 'vgg_best_model.pt')

    # 注意 如果设备 带不动  把  num_workers 设置为 0 试一试 ！！！
    # 出现警告请忽略， 只要不报错， 就等待 程序运行完  就行， 耗时约30分钟（跟自己设备相关）



