# kmeans_pytorch_new
kmeans_pytorch是支持GPU的pytroch写的kmeans包，但官网文档叙述不清晰，重新加了点注释
[原包链接](https://pypi.org/project/kmeans-pytorch/)

# 笔记补充：
# 浅析kmeans_torch库中的initialize与距离函数：
```py
import torch
import numpy as np

def initialize(X: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    选出中心点
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)  # X的行数

    indices = np.random.choice(num_samples, num_clusters, replace=False)
    # 从num_samples（一维） 中随机不重复
    # 取num_clusters个数，组成一个一维array —— indices
    initial_state = X[indices]  # 将X的indices这些行取出来，作为一个新的tensor
    return initial_state

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N1*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N2*M
    B = data2.unsqueeze(dim=0)

    # 因为有上面两个unsqueeze操作，使得A中的每一行变为单独一个维度，然后对应全部的B，利用广播机制进行减法运算
    # 最后A-B返回一个N1*N2*M的tensor

    dis = (A - B) ** 2.0  # 点乘：对应位置分别相乘，仍然为一个N1*N2*M的tensor

    # return N1*N2 matrix for pairwise distance
    # 倒数第一维度：最内层的维度：M维（2维）彼此相加,这个squeeze()貌似没作用
    dis = dis.sum(dim=-1).squeeze()
    return dis  # 返回一个N1*N2的tensor，每一行分别代表某点对N2个中心点的距离的平方和

```

```py
data_size, dims, num_clusters = 10, 2, 3
x = np.random.randn(data_size, dims) / 10  # 除以10是为了缩小数值，方便画图
x = torch.from_numpy(x)
print(x)
b = initialize(x, 3)
print(b)
```

结果：
![img](https://img2023.cnblogs.com/blog/2635041/202303/2635041-20230324214317862-2081254914.png)
x共10行表示10个点，每行两列，分别表示横坐标和纵坐标
b 为x中随机选取的3个点

## 经过unsqueeze操作
观察求距离的代码，发现有unsqueeze操作，如果没有这个操作，那么tensor x与b由于维数不同而没法用广播机制且没法进行减法运算
经过unsqueeze后，可以近似认为，x中的每一行成了一维，b整个成了一维，然后x-b就可以利用广播机制进行运算：
![img](https://img2023.cnblogs.com/blog/2635041/202303/2635041-20230324214853091-83139640.png)
x-b就是10\*3\*2的tensor
![img](https://img2023.cnblogs.com/blog/2635041/202303/2635041-20230324215014422-169455792.png)
然后又进行了sum操作：
对`dim=-1`进行`sum`,就等价于对`dim=2`进行`sum`，也就是M维度
测试：
```py
c = x - b
print(c.shape)
c = c.sum(dim=-1)
print(c.shape)
print(c)
c = c.squeeze()
print(c)
print(c.shape)
```
结果：
![img](https://img2023.cnblogs.com/blog/2635041/202303/2635041-20230324215346384-1674188593.png)
