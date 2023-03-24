import numpy as np
import torch
from tqdm import tqdm

"""
X 的格式：
每一行代表一个点，后面有两列：
    第0列：该点的x坐标
    第1列：该点的y坐标
"""


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


def kmeans(
        X: torch.Tensor,
        num_clusters: int,
        distance: str = 'euclidean',
        tol: float = 1e-4,
        device: torch.device = torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters  簇
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']  欧几里得距离，余弦距离
    :param tol: (float) threshold [default: 0.0001]    界，我们判定是否结束的阈值
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    # 算距离
    if distance == 'euclidean':  # 欧几里得距离
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':  # 余弦距离
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float， 转化为浮点数
    X = X.float()

    # transfer to device  转化设备
    X = X.to(device)

    # initialize，随机选出中心点
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)

        # 选出dis中每一行的最小值的索引index，即找出该点距离最近的中心点
        choice_cluster = torch.argmin(dis, dim=1)  # n*1的tensor(共n个点)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(
                choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X: torch.Tensor,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    # 算距离
    if distance == 'euclidean':  # 欧几里得记录
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':  # 余弦距离
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float，转化为浮点数
    X = X.float()

    # transfer to device，转化设备
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


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


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis
