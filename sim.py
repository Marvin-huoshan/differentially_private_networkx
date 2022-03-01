import networkx as nx
from main import connect_integers
import pickle

def avg_cluster(G):
    '''计算图的平均聚类系数'''
    return nx.average_clustering(G)

def avg_shortest_path(G):
    '''计算图的平均最短路径长度'''
    return nx.average_shortest_path_length(G)

def compare(G, G_file):
    '''
    对修改前后图的全局图可用性指标进行比较
    :param G:修改前的图
    :param G_file:修改后的图文件
    :return:
    '''
    with open(G_file,'rb') as file_to_read:
        G_result = pickle.load(file_to_read)
    G_result = nx.from_numpy_matrix(G_result)
    print('G_connect edges:', nx.number_of_edges(G))
    print('G_modify edges:', nx.number_of_edges(G_result))
    print('G_connect nodes:', nx.number_of_nodes(G))
    print('G_modify nodes:', nx.number_of_nodes(G_result))
    print('G_connect AVE:', 2 * nx.number_of_edges(G) / nx.number_of_nodes(G))
    print('G_modify AVE:', 2 * nx.number_of_edges(G_result) / nx.number_of_nodes(G_result))
    print('G_connect ACC:', avg_cluster(G))
    print('G_modify ACC:', avg_cluster(G_result))
    list_G = list(max(nx.connected_components(G_result)))
    G_result = nx.subgraph(G_result, list_G)
    print('G_connect APL:', avg_shortest_path(G))
    print('G_modify APL:', avg_shortest_path(G_result))

if __name__ == '__main__':
    G_face = nx.read_edgelist('dataset/facebook_combined.txt')
    G_face_connect = connect_integers(G_face)
    #print('add node!')
    #compare(G_face_connect, 'pkfile/face/face_1_noise_group.pk')
    compare(G_face_connect, 'pkfile/face_re_matrix.pk')