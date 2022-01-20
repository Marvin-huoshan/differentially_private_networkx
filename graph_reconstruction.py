import time

import networkx as nx
import numpy as np
from scipy import sparse
from tqdm import tqdm
import pickle
from main import connect_integers
from pandas import *

def domain_set(G):
    #求出图G的支配集
    return list(nx.dominating_set(G))

def is_domained(G, domain_set):
    #求出图G的被支配节点集
    nodes_all = nx.nodes(G)
    return list(set(nodes_all).difference(set(domain_set)))

def edge_betweenness_centrality(G, sources, targets):
    #求出原始边的介数中心性
    return nx.edge_betweenness_centrality_subset(G=G, sources=sources, targets=targets)

def edge_betweenness_centrality_all(G):
    #求出原始边的介数中心性
    return nx.edge_betweenness_centrality(G=G)

def betweenness2matrix(dicts, name, length):
    #将betweenness转为矩阵存储
    betweenness_pair = list(dicts.keys())
    betweenness_value = list(dicts.values())
    row = [i[0] for i in betweenness_pair]
    col = [i[1] for i in betweenness_pair]
    adj = sparse.coo_matrix((betweenness_value, (row, col)), shape=(length, length)).tolil()
    savepath = 'pkfile' + '/'
    with open(savepath + name + '.pk', 'wb') as file_to_write:
        pickle.dump(adj, file_to_write)
    return adj

if __name__ == '__main__':
    G_face = nx.read_edgelist('dataset/facebook_combined.txt')
    #G_Email = nx.read_edgelist('dataset/Email-Enron.txt')
    #G_cond = nx.read_edgelist('dataset/CA-CondMat.txt')
    #G_dblp = nx.read_edgelist('dataset/com-dblp.ungraph.txt')
    #最大连通子图，并将节点进行编号
    G_face_connect = connect_integers(G_face)
    #G_Email_connect = connect_integers(G_Email)
    #G_cond_connect = connect_integers(G_cond)
    #G_dblp_connect = connect_integers(G_dblp)
    #求出每一个图的支配集
    face_domain = domain_set(G_face_connect)
    #Email_domain = domain_set(G_Email_connect)
    #cond_domain = domain_set(G_cond_connect)
    #dblp_domain = domain_set(G_dblp_connect)
    #求出每一个图的被支配集
    face_isdomained = is_domained(G_face_connect, face_domain)
    #Email_isdomained = is_domained(G_Email_connect, Email_domain)
    #cond_isdomained = is_domained(G_cond_connect, cond_domain)
    #dblp_isdomained = is_domained(G_dblp_connect, dblp_domain)
    #求出原始图中支配集与被支配集之间连边的介数中心性，将其保存在稀疏矩阵中
    face_edge_reconstruct = edge_betweenness_centrality(G_face_connect, face_domain, face_isdomained)
    betweenness2matrix(face_edge_reconstruct, 'face_betweenness_matrix', nx.number_of_nodes(G_face_connect))
    exit(0)
    Email_edge_reconstruct = edge_betweenness_centrality(G_Email_connect, Email_domain, Email_isdomained)
    print(Email_edge_reconstruct)
    cond_edge_reconstruct = edge_betweenness_centrality(G_cond_connect, cond_domain, cond_isdomained)
    print(cond_edge_reconstruct)
    dblp_edge_reconstruct = edge_betweenness_centrality(G_dblp_connect, dblp_domain, dblp_isdomained)
    print(dblp_edge_reconstruct)
