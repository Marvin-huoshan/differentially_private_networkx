import random

import networkx as nx
import numpy as np
import scipy as sp

from graph_reconstruction import *

def data_ready(G, file, matrix_file, name2):
    #将加完噪声的数据、原始数据、骨干子网络准备好
    noise_group = pickle.load(open(file, 'rb'))
    noise_group = [a for b in noise_group for a in b]
    G_dict = dict(sorted(nx.degree(G), key=lambda x: x[1], reverse=True))
    G_key = list(G_dict.keys())
    print(G_key)
    G_degree = list(G_dict.values())
    G_degree_part = G_degree[263:]
    #加完噪声后的度分布
    noise_group.extend(G_degree_part)
    #控制集
    G_domain = domain_set(G)
    add_edges(G, noise_group, matrix_file, G_key, name2)

def add_edges(G, noise_group, matrix_file, G_key, name2):
    #按照加完噪声的度序列，根据节点与控制集之间
    adj = nx.adjacency_matrix(G)
    adj = sparse.lil_matrix(adj)
    print(noise_group)
    print(sum(noise_group)/2)
    with open(matrix_file,'rb') as file_to_read:
        matrix = pickle.load(file_to_read)
    nodes_number = nx.number_of_nodes(G)
    re_matrix = np.zeros((nodes_number, nodes_number))
    for i in tqdm(range(len(noise_group)), desc='按照betweenness_subset进行第一次加边'):
        #根据在原始图中的边介中心性由大到小进行加边
        for j in range(noise_group[i]):
            col_argmax = [matrix.getrow(i).A.argmax()]
            col_argmax = col_argmax.pop()
            if noise_group[col_argmax] == -100:
                matrix[i, col_argmax] = 0
                matrix[col_argmax, i] = 0
                continue
            #在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            re_matrix[col_argmax, i] = 1
            #边介中心性矩阵对应位置赋值为0
            matrix[i, col_argmax] = 0
            matrix[col_argmax, i] = 0
            #noise_group对应位置减1
            noise_group[i] -= 1
            noise_group[col_argmax] -= 1
            if noise_group[i] == 0 or matrix.getrow(i).A.max() == 0:
                break
    key_two = [G_key[i] for i in range(len(G_key)) if noise_group[i] != 0]
    print(noise_group)
    print(re_matrix.sum() / 2)
    print(sum(noise_group)/2)
    #matrix_all = nx.edge_betweenness_centrality_subset(G=G, sources=key_two, targets=G_key)
    #matrix_all = betweenness2matrix(matrix_all, name2, nx.number_of_nodes(G))
    with open('pkfile/face_betweenness_matrix2.pk', 'rb') as file_to_read:
        matrix_all = pickle.load(file_to_read)
    for i in tqdm(range(len(noise_group)), desc='按照betweenness进行第二次加边'):
        #根据在原始图中的边介中心性由大到小进行加边
        for j in range(noise_group[i]):
            col_argmax = [matrix_all.getrow(i).A.argmax()]
            col_argmax = col_argmax.pop()
            #条件2：在之前已经加过边，noise_group不应该重复减
            if noise_group[col_argmax] == -100 or re_matrix[i, col_argmax] == 1:
            #if re_matrix[i, col_argmax] == 1:
                matrix_all[i, col_argmax] = 0
                matrix_all[col_argmax, i] = 0
                continue
            # 在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            re_matrix[col_argmax, i] = 1
            # 边介中心性矩阵对应位置赋值为0
            matrix_all[i, col_argmax] = 0
            matrix_all[col_argmax, i] = 0
            # noise_group对应位置减1
            noise_group[i] -= 1
            noise_group[col_argmax] -= 1
            if noise_group[i] == 0 or matrix_all.getrow(i).A.max() == 0:
                break
    print(noise_group)
    print(re_matrix.sum() / 2)
    print(sum(noise_group) / 2)
    #获得二跳以及三跳可达矩阵
    A2, A3, A4 = matrix_arrive(adj)
    matrix_add(noise_group, A2, A3, A4, re_matrix)


def matrix_arrive(matrix_all):
    # 使用二跳和三跳可达矩阵来进行加边
    AS = matrix_all.tocsr()
    A2S = AS * AS
    # A22S = sp.coo_matrix(A2S)
    A3S = A2S * AS
    A4S = A3S * AS
    # A33S = sp.coo_matrix(A3S)
    # 非零值归1
    print('zero->1')
    AS[np.nonzero(AS)] = 1
    A2S[np.nonzero(A2S)] = 1
    A3S[np.nonzero(A3S)] = 1
    A4S[np.nonzero(A4S)] = 1
    # n跳矩阵
    print('mkdir n-hop-matrix')
    A2 = A2S - AS
    A3 = A3S - A2S
    A4 = A4S - A3S
    # 非1值归0
    print('none 1 -> zero')
    A2 = A2.tolil()
    A3 = A3.tolil()
    A4 = A4.tolil()
    A2[A2 != 1] = 0
    A3[A3 != 1] = 0
    A4[A4 != 1] = 0
    # 对角线归0
    print('diag -> 0')
    row, col = np.diag_indices_from(A2)
    A2[row, col] = 0
    row, col = np.diag_indices_from(A3)
    A3[row, col] = 0
    row, col = np.diag_indices_from(A4)
    A4[row, col] = 0
    return A2, A3, A4

def test(matrix_file):
    with open(matrix_file,'rb') as file_to_read:
        matrix = pickle.load(file_to_read)
    list1 = matrix.getrow(1).A
    list1 = [a for b in list1 for a in b]
    print(list1)
    print(list1.index(0))

def matrix_add(noise_group, A2, A3, A4, re_matrix):
    #使用二跳可达矩阵进行加边
    for i in tqdm(range(len(noise_group)), desc='按照二跳可达矩阵进行第三次加边'):
        for j in range(noise_group[i]):
            col_argmax = [A2.getrow(i).A.argmax()]
            col_argmax = col_argmax.pop()
            if noise_group[col_argmax] <= -100:
                A2[i, col_argmax] = 0
                continue
            # 在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            re_matrix[col_argmax, i] = 1
            # 二跳可达矩阵对应位置赋值为0
            A2[i, col_argmax] = 0
            A2[col_argmax, i] = 0
            # noise_group对应位置减1
            noise_group[i] -= 1
            noise_group[col_argmax] -= 1
            if noise_group[i] == 0 or A2.getrow(i).A.max() == 0:
                break
    print(noise_group)
    print(re_matrix.sum() / 2)
    print(sum(noise_group) / 2)
    # 使用三跳可达矩阵进行加边
    for i in tqdm(range(len(noise_group)), desc='按照三跳可达矩阵进行第三次加边'):
        for j in range(noise_group[i]):
            col_argmax = [A3.getrow(i).A.argmax()]
            col_argmax = col_argmax.pop()
            if noise_group[col_argmax] <= -50:
                A3[i, col_argmax] = 0
                continue
            # 在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            re_matrix[col_argmax, i] = 1
            # 二跳可达矩阵对应位置赋值为0
            A3[i, col_argmax] = 0
            A3[col_argmax, i] = 0
            # noise_group对应位置减1
            noise_group[i] -= 1
            noise_group[col_argmax] -= 1
            if noise_group[i] == 0 or A3.getrow(i).A.max() == 0:
                break
    print(noise_group)
    print(re_matrix.sum() / 2)
    print(sum(noise_group) / 2)
    # 使用三跳可达矩阵进行加边
    for i in tqdm(range(len(noise_group)), desc='按照四跳可达矩阵进行第四次加边'):
        for j in range(noise_group[i]):
            col_argmax = [A4.getrow(i).A.argmax()]
            col_argmax = col_argmax.pop()
            if noise_group[col_argmax] <= -20:
                A4[i, col_argmax] = 0
                continue
            # 在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            re_matrix[col_argmax, i] = 1
            # 二跳可达矩阵对应位置赋值为0
            A4[i, col_argmax] = 0
            A4[col_argmax, i] = 0
            # noise_group对应位置减1
            noise_group[i] -= 1
            noise_group[col_argmax] -= 1
            if noise_group[i] == 0 or A4.getrow(i).A.max() == 0:
                break
    print(noise_group)
    print(re_matrix.sum() / 2)
    print(sum(noise_group) / 2)
    savepath = 'pkfile' + '/'
    with open(savepath + 'face_re_matrix' + '.pk', 'wb') as file_to_write:
        pickle.dump(re_matrix, file_to_write)
    with open(savepath + 'noise_group_random' + '.pk', 'wb') as file_to_write:
        pickle.dump(noise_group, file_to_write)

def random_nodes(G, noise_group, re_matrix):
    #对于待加边序列，加入max(noise_group)个节点，并与其他节点进行随机连边
    with open(noise_group, 'rb') as file_to_read:
        noise_group = pickle.load(file_to_read)
    with open(re_matrix, 'rb') as file_to_read:
        re_matrix = pickle.load(file_to_read)
    graph = nx.from_numpy_matrix(re_matrix)
    added_list = []
    for i in range(max(noise_group)):
        added_list.append(str(i + 1) + '-add')
        graph.add_node(str(i + 1) + '-add')
    for i in range(len(noise_group)):
        if noise_group[i] > 0:
            random.shuffle(added_list)
            for j, k in zip(range(noise_group[i]), added_list):
                graph.add_edge(i, k)
                print('add edge:', str(i), ',', str(k))
                noise_group[i] -= 1
    face_1_noise_group = nx.to_numpy_matrix(graph)
    print(face_1_noise_group)
    savepath = 'pkfile' + '/' + 'face/'
    with open(savepath + 'face_1_noise_group' + '.pk', 'wb') as file_to_write:
        pickle.dump(face_1_noise_group, file_to_write)

if __name__ == '__main__':
    G = nx.read_edgelist('dataset/facebook_combined.txt')
    G_connect = connect_integers(G)
    data_ready(G_connect, 'pict/face_noise_group.pk', 'pkfile/face_betweenness_matrix.pk', 'face_betweenness_matrix2')
    #random_nodes(G_connect, 'pkfile/noise_group_random.pk', 'pkfile/face_re_matrix.pk')