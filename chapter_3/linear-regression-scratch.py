
import random
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    """
        构建测试数据集 生成y=Xw+b+噪声
    :param
        w: n 维向量 权重
        b: 标准偏差
        num_examples: 数据量
    :return
        X: n 维向量
        Y:
    """

    # 创建一个形状为 (num_examples 样本数, len(w) 样本特征数) 的张量 X， 从均值为 0，标准差为 1 的正态分布中随机采样得到
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 表示对 X 和 w 进行矩阵乘法操作，得到一个形状为 (num_examples, 1) 的张量。
    # 然后，将形状为 (num_examples, 1) 的张量和形状为 (1,) 的张量 b 相加，得到一个形状为 (num_examples, 1) 的张量 y。
    y = torch.matmul(X, w) + b
    # 向 y 张量中加入了一个均值为 0，标准差为 0.01 的噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 小批量 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 包含 0 到 num_examples-1 的整数 索引
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    # 1. 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    # 2. 小批量数据
    batch_size = 10

    # 3. 初始化 模型参数 （ y = <X,w> + b ）
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(w, '\n', b)

    # 训练
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # net(X, w, b)：将样本 X 和 初始化的模型参数(权重、偏差) 带入线性方程，计算出的结果 与 y(进行损失计算)
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            # 首先，计算当前小批量数据集的损失 $l$，即 $l = loss(net(X, w, b), y)$。
            # 然后，对 $l$ 调用 sum() 方法，将当前小批量数据集的损失求和得到一个标量张量。
            # 最后，调用 backward() 方法，自动计算当前小批量数据集的损失相对于模型参数 w 和 b 的梯度，并将梯度累加到相应的张量的 grad 属性中。
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数，也就是 [w, b]
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1},[w, b] {[w, b]}, loss {float(train_l.mean()):f}')
