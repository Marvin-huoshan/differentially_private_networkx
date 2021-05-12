import networkx as nx
import xlsxwriter
import matplotlib as plt
import xlrd
import numpy as np
from scipy.optimize import root

def func(G,name,file):
    workbook = xlrd.open_workbook(file)
    mySheet = workbook.sheet_by_name('Sheet1')
    cont_list = mySheet.col_values(1)
    cont_list.pop(0)
    set_cont_list = list(set(cont_list))
    sorted_set_cont_list = list(sorted(set(cont_list)))
    mid = sorted_set_cont_list[len(sorted_set_cont_list)//2]
    #print(cont_list)
    sorted_degree_dict = dict(sorted(nx.degree(G), key=lambda x: x[1], reverse=True))
    # list1 = list(sorted(set(sorted_degree_dict.values()), reverse=True))
    #y = list(sorted(set(sorted_degree_dict.values()), reverse=True))
    x = list(sorted(set(sorted_degree_dict.values())))
    y = cont_list[:]
    y.reverse()
    index = y.index(mid)
    x = x[index+1:]
    y = y[index+1:]
    #y =
    #y = list(sorted(sorted_degree_dict.values(), reverse=True))
    #x = [i for i in range(1, len(y) + 1)]
    f1 = np.polyfit(x, y, 5)
    #print('f1 is:', f1)
    p1 = np.poly1d(f1)
    #print('p1 is :\n', p1)
    yvals = p1(x)  # 拟合y值
    p2 = np.polyder(p1, 1)#一阶导数
    #print('p2 is:\n', p2)
    p3 = np.polyder(p1,2)
    yvals1 = p2(x)
    yvals2 = p3(x)

    #print('yvals is :\n', yvals)
    # 绘图
    plt.figure(figsize=(20,10))
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    #plot3 = plt.plot(x, yvals1,'g',label = 'polyder values')
    #plot3 = plt.plot(x, yvals2,'y',label = 'polyder2 values')
    my_x_ticks = np.arange(0, 140, 5)
    plt.xticks(my_x_ticks)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title('polyfitting of ' + name)
    solution2 = root(fun=p3, x0=[1,2,3])
    solution = root(fun=p2, x0=[1,30,50,75,85])
    #list_2 = list[i for i in list(p3(solution.x))]
    list_2 = []

    for i in range(len(solution.x)):
        if p3(solution.x[i]) > 0:
            list_2.append([solution.x[i],p3(solution.x[i])])
    sorted_list = sorted(list_2,key=lambda x:x[1],reverse=True)
    print(sorted_list)
    path = '/Users/mac/Desktop/networkx/'
    plt.savefig(path + '/' + name + '.png')
    plt.show()
    #print(solution.x[0])
    #print(solution2.x[0])
    #print(p1(solution.x[0]))
    #print(p1(solution.x[0]))
    #print(solution.x)
    #print(p1(solution.x[0]))
    #print(solution2.x[0])
    #print(p1(solution2.x[0]))
    return p1(solution2.x[0])

def same_degree(G,name):
    '''

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
    G_face_connect = connect_integers(G_face)
    same_degree(G_face_connect,'face')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
