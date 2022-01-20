import networkx as nx
import numpy as np

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
    print(noise_group)
    with open(matrix_file,'rb') as file_to_read:
        matrix = pickle.load(file_to_read)
    nodes_number = nx.number_of_nodes(G)
    re_matrix = np.zeros((nodes_number, nodes_number))
    for i in tqdm(range(len(noise_group)), desc='按照betweenness_subset进行第一次加边'):
        #根据在原始图中的边介中心性由大到小进行加边
        for j in range(noise_group[i]):
            col_argmax = [matrix.getrow(i).A.argmax()]
            #在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            #边介中心性矩阵对应位置赋值为0
            matrix[i, col_argmax] = 0
            #noise_group对应位置减1
            noise_group[i] -= 1
            if noise_group[i] == 0 or matrix.getrow(i).A.max() == 0:
                break
    key_two = [G_key[i] for i in range(len(G_key)) if noise_group[i] != 0]
    print(noise_group)
    #matrix_all = nx.edge_betweenness_centrality_subset(G=G, sources=key_two, targets=G_key)
    #matrix_all = betweenness2matrix(matrix_all, name2, nx.number_of_nodes(G))
    with open('pkfile/face_betweenness_matrix2.pk', 'rb') as file_to_read:
        matrix_all = pickle.load(file_to_read)
    for i in tqdm(range(len(noise_group)), desc='按照betweenness进行第二次加边'):
        #根据在原始图中的边介中心性由大到小进行加边
        for j in range(noise_group[i]):
            col_argmax = [matrix_all.getrow(i).A.argmax()]
            # 在矩阵对应位置加边
            re_matrix[i, col_argmax] = 1
            # 边介中心性矩阵对应位置赋值为0
            matrix[i, col_argmax] = 0
            # noise_group对应位置减1
            noise_group[i] -= 1
            if noise_group[i] == 0 or matrix_all.getrow(i).A.max() == 0:
                break
    print(noise_group)


if __name__ == '__main__':
    G = nx.read_edgelist('dataset/facebook_combined.txt')
    G_connect = connect_integers(G)
    data_ready(G_connect, 'pict/face_noise_group.pk', 'pkfile/face_betweenness_matrix.pk', 'face_betweenness_matrix2')