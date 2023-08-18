import torch
from IPython import display
from d2l import torch as d2l

# 1. 加载数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2. 初始化模型参数
num_inputs = 784  # 本节将展平每个图像，把它们看作长度为784的向量。 1*28*28 = 784
num_outputs = 10  # 分类数量

# 784 * 10 的矩阵
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 1 * 10 的行向量
b = torch.zeros(num_outputs, requires_grad=True)


# 3. 定义 softmax 操作
def softmax(X):
    """
    实现softmax由三个步骤组成：
    对每个项求幂（使用exp）；
    对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
    将每一行除以其规范化常数，确保结果的和为1。
    示例：
        2个样本（5个特征值），softmax 之后，每个样本 特征值概率之和是 1
    tensor([[-0.6035,  0.2890, -0.9154, -0.1194, -0.5019],
        [ 0.9122,  0.5518, -3.0488, -0.1056, -0.2067]])
    tensor([[0.1449, 0.3536, 0.1061, 0.2351, 0.1604],
            [0.4159, 0.2900, 0.0079, 0.1503, 0.1358]])
    tensor([1.0000, 1.0000])
    """

    X_exp = torch.exp(X)
    # X.sum(0, keepdim=True)： 列维度求和，得到一行
    # X.sum(1, keepdim=True)： 行维度求和，得到一列
    # partition 就是转成 exp 之后的 sum
    partition = X_exp.sum(1, keepdim=True)
    # print(partition)
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    """
    """
    # W 是 784 * 10 的矩阵，所以 W.shape[0] = 784
    # 将 X 转成 784 行的张量
    x_reshape = X.reshape((-1, W.shape[0]))
    # X.shape = torch.Size([256, 1, 28, 28])
    # x_reshape.shape = torch.Size([256, 784])
    # print(X.shape)
    # print(x_reshape.shape)
    # [256 * 784] x [784 * 10] = [256 * 10]  + [1 * 10]
    return softmax(torch.matmul(x_reshape, W) + b)


def cross_entropy(y_hat, y):
    """
    交叉熵损失函数
    :param y_hat: 预测概率
    :param y: 真实类型
    :return:
    """
    # `y_hat[range(len(y_hat)), y]` 表示从 `y_hat` 中取出所有样本预测为真实标签的预测概率，
    # 使用`torch.log()`计算其对数，最后使用负号取反得到损失值。
    # 损失函数要求导，最好化成加减，所以对表达式，取 log
    return - torch.log(y_hat[range(len(y_hat)), y])


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    # X = torch.normal(0, 1, (2, 5))
    # print(X)
    # X_prob = softmax(X)
    # X_prob, X_prob.sum(1)
    # print(X_prob)
    # print(X_prob.sum(1))
    # print("-------")
    #
    # W = torch.normal(0, 0.01, size=(2, 5), requires_grad=True)
    # print(W)
    # x_reshape = X.reshape((-1, W.shape[0]))
    # print(x_reshape)

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6],
                          [0.3, 0.2, 0.5]])
    print(y_hat.shape)
    # [0, 1] 表示要取出的行的索引，y 表示要取出的列的索引
    # 在 y_hat[[0, 1], y] 中，[0, 1] 表示要取出的行的索引，即第 $0$ 行和第 $1$ 行。
    # 而 y 是要取出的列的索引，它的含义是使用真实标签 y 作为列的索引，从 y_hat 中取出对应的预测概率。
    print(y_hat[[0, 1], y])
    # print(y_hat[[0, 1], [1, 2]])
    # print(y_hat[[0, 1], [0, 2]])

    print(cross_entropy(y_hat, y))
    # num_epochs = 10
    # train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
