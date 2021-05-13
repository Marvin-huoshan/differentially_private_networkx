import networkx as nx
import xlsxwriter
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from scipy.optimize import root
import pandas as pd
import os

def func(G,name,file):
    '''
    使用截断机制
    :param G:
    :param name:
    :param file:
    :return:
    '''
    #读取第一列的值
    df_y = pd.read_excel(file,usecols=[1])
    df_list = df_y.values.tolist()
    #度数从大到小出现的次数
    cont_list = [a for b in df_list for a in b]
    #cont_list.reverse()
    print(cont_list)
    df_x = pd.read_excel(file,usecols=[0])
    df_list = df_x.values.tolist()
    degree_list = [a for b in df_list for a in b]
    max = degree_list[0]
    #degree_list.reverse()
    print(degree_list)
    mid = degree_list[len(degree_list)//2]
    #print(cont_list)
    sorted_degree_dict = dict(sorted(nx.degree(G), key=lambda x: x[1], reverse=True))
    # list1 = list(sorted(set(sorted_degree_dict.values()), reverse=True))
    #y = list(sorted(set(sorted_degree_dict.values()), reverse=True))
    #x = list(sorted(set(sorted_degree_dict.values())))
    x = degree_list[:]
    y = cont_list[:]
    #y.reverse()
    '''index = x.index(mid)
    x = x[:index+1]
    y = y[:index+1]
    #y ='''
    f10 = np.polyfit(x, y, 10)
    #9次曲线拟合
    f9 = np.polyfit(x, y, 9)
    #8次曲线拟合
    f8 = np.polyfit(x, y, 8)
    #7次曲线拟合
    f7 = np.polyfit(x, y, 7)
    #6次曲线拟合
    f6 = np.polyfit(x, y, 6)
    f5 = np.polyfit(x, y, 5)
    #print('f1 is:', f1)
    #得到拟合的多项式，按照阶数从高到低
    #p10 = np.poly1d(f10)
    p9 = np.poly1d(f9)
    p8 = np.poly1d(f8)
    p7 = np.poly1d(f7)
    p6 = np.poly1d(f6)
    p5 = np.poly1d(f5)
    #print('p1 is :\n', p1)
    yvals = p9(x)  # 拟合曲线的y值
    #p10_1 = np.polyder(p10, 1)
    p9_1 = np.polyder(p9, 1)#一阶导数
    p8_1 = np.polyder(p8, 1)
    p7_1 = np.polyder(p7, 1)
    p6_1 = np.polyder(p6, 1)
    p5_1 = np.polyder(p5, 1)
    #print('p2 is:\n', p2)
    p1 = p9
    p3 = np.polyder(p1,2)
    #yvals10 = p10_1(x)
    yvals9 = p9_1(x)
    yvals8 = p8_1(x)
    yvals7 = p7_1(x)
    yvals6 = p6_1(x)
    yvals5 = p5_1(x)
    #yvals2 = p3(x)

    #print('yvals is :\n', yvals)
    # 绘图
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(121)
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.gca().invert_xaxis()
    ax1.set_title('polyfitting of ' + name)
    ax1.set_xlabel('degree')
    ax1.set_ylabel('Frequency')
    ax1.legend(fontsize='large', loc='upper left')
    ax2 = plt.subplot(122)
    #plot10 = plt.plot(x,yvals10,'m',label='polyder10 values')
    plot9 = plt.plot(x, yvals9,'y',label = 'polyder9 values')
    plot8 = plt.plot(x, yvals8, 'r', label='polyder8 values')
    plot7 = plt.plot(x, yvals7, 'b', label='polyder7 values')
    plot6 = plt.plot(x, yvals6, 'g', label='polyder6 values')
    plot5 = plt.plot(x, yvals5, 'c', label='polyder5 values')
    plt.gca().invert_xaxis()
    #plot3 = plt.plot(x, yvals2,'y',label = 'polyder2 values')
    ax2.set_title('polyder1')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(fontsize='large', loc='upper right')
    my_x_ticks = np.arange(0, max, 100)
    plt.xticks(my_x_ticks)
    plt.legend(loc=4)  # 指定legend的位置右下角
    p2 = p9_1
    solution2 = root(fun=p3, x0=[1,2,3])
    solution = root(fun=p2, x0=[1,30,50,75,85])
    #list_2 = list[i for i in list(p3(solution.x))]
    list_2 = []

    for i in range(len(solution.x)):
        if p3(solution.x[i]) > 0:
            list_2.append([solution.x[i],p3(solution.x[i])])
    sorted_list = sorted(list_2,key=lambda x:x[1],reverse=True)
    print(sorted_list)
    savepath = 'pict' + '/' + name
    isExists = os.path.exists(savepath)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(savepath)
        print(savepath + ' 创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(savepath + ' 目录已存在')
    plt.savefig(savepath + '.png')
    plt.show()
    return p1(solution2.x[0])


def degree_cout(G,name):
    '''
    每个度出现的次数
    :param G:Graph
    :param name: xlsx'name
    :return: none
    '''

    sorted_degree_dict = dict(sorted(nx.degree(G),key = lambda x:x[1],reverse=True))
    list1 = list(sorted(set(sorted_degree_dict.values()),reverse=True))
    list2 = list(sorted_degree_dict.keys())
    cont = 0
    conts = 0
    workbook = xlsxwriter.Workbook(name + '-degree_cont.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,'degree')
    worksheet.write(0,1,'个数')
    worksheet.write(0,2,'百分比')
    worksheet.write(1,0, str(list1[cont]))
    #worksheet.write(1,1,str(list2[cont]))
    for key, value in sorted_degree_dict.items():
        if value == list1[cont]:
            #worksheet.write(cont+1,conts,key)
            conts += 1
            #print(conts)
            continue
        worksheet.write(cont+1,1,conts)
        worksheet.write(cont+1,2,'{:.4%}'.format(conts/nx.number_of_nodes(G)))
        if cont < len(list1)-1:
            cont += 1
            worksheet.write(cont+1,0,str(list1[cont]))
            #worksheet.write(cont+1,1,key)
            conts = 1
        print(key)
    worksheet.write(cont + 1, 1, conts)
    worksheet.write(cont + 1, 2, '{:.4%}'.format(conts / nx.number_of_nodes(G)))
    workbook.close()

def same_degree(G,name):
    '''
    度值以及对应的节点id
    :param G:Graph
    :param name: xlsx'name
    :return: none
    '''

    sorted_degree_dict = dict(sorted(nx.degree(G),key = lambda x:x[1],reverse=True))
    list1 = list(sorted(set(sorted_degree_dict.values()),reverse=True))
    list2 = list(sorted_degree_dict.keys())
    cont = 0
    conts = 1
    workbook = xlsxwriter.Workbook(name + '-degree.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,'degree')
    worksheet.write(1,0, str(list1[cont]))
    worksheet.write(1,1,str(list2[cont]))
    for key, value in sorted_degree_dict.items():
        if value == list1[cont]:
            worksheet.write(cont+1,conts,key)
            conts += 1
            continue
        if cont < len(list1)-1:
            cont += 1
            worksheet.write(cont+1,0,str(list1[cont]))
            worksheet.write(cont+1,1,key)
            conts = 2
        print(key)
    workbook.close()

def connect_integers(G):
    '''
    处理传入的图，获取其最大连通子图，并将节点编号
    :param G:
    :return:
    '''
    G = nx.convert_node_labels_to_integers(G)
    list_G = list(max(nx.connected_components(G)))
    G_connect = nx.subgraph(G, list_G)
    G_face_connect = nx.convert_node_labels_to_integers(G_connect)
    return G_face_connect

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    G_face = nx.read_edgelist('facebook_combined.txt')
    G_Email = nx.read_edgelist('Email-Enron.txt')
    G_cond = nx.read_edgelist('CA-CondMat.txt')
    G_dblp = nx.read_edgelist('com-dblp.ungraph.txt')
    G_face_connect = connect_integers(G_face)
    G_Email_connect = connect_integers(G_Email)
    G_cond_connect = connect_integers(G_cond)
    G_dblp_connect = connect_integers(G_dblp)
    #degree_cout(G_Email_connect, 'Email')
    #degree_cout(G_cond_connect,'cond')
    #degree_cout(G_dblp_connect,'dblp')
    func(G_face_connect,'face','face-degree_cont.xlsx')
    func(G_Email_connect,'Email','Email-degree_cont.xlsx')
    func(G_cond_connect,'cond','cond-degree_cont.xlsx')
    func(G_dblp_connect,'dblp','dblp-degree_cont.xlsx')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
