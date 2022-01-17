import networkx as nx
import xlsxwriter
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from scipy.optimize import root
import pandas as pd
import os
import math
import pickle
from multiprocessing import Pool,Process

def func1(name,file,break_point_low,break_point_high):
    '''
    使用截断机制,使用度作为y，x轴为元素数量
    :param G:
    :param name:
    :param file:
    :param break_point: 截断点
    :return:
    '''
    #读取第一列的值
    df_y = pd.read_excel(file,usecols=[0])
    df_list = df_y.values.tolist()
    cont_list = [a for b in df_list for a in b]
    cont_list = [i for i in cont_list if i > break_point_low and i < break_point_high]
    df_x = [i for i in range(len(cont_list))]
    x = df_x[:]
    y = cont_list[:]
    f12 = np.polyfit(x, y, 12)
    p12 = np.poly1d(f12)
    yvals = p12(x)  # 拟合曲线的y值
    p12_1 = np.polyder(p12, 1)#一阶导数
    p12_2 = np.polyder(p12, 2)#二阶导数
    '''solution = root(fun=p12_2, x0=[157,156,148,135,120,103,85,67,46,36,18,14])
    print(solution.x)
    print(p12(solution.x))'''
    print('零点:',p12_2.r)
    print('degree:',p12(p12_2.r))
    yvals12 = p12_1(x)
    yvals12_2 = p12_2(x)
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(121)
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.gca().invert_xaxis()
    ax1.set_title('polyfitting of ' + name)
    ax1.set_xlabel('x')
    ax1.set_ylabel('degree')
    ax1.legend(fontsize='large', loc='upper right')
    ax1.invert_xaxis()
    ax2 = plt.subplot(122)
    plot9 = plt.plot(x, yvals12,'y',label = 'polyder1 values')
    plot8 = plt.plot(x, yvals12_2, 'r', label='polyder2 values')
    plt.gca().invert_xaxis()
    ax2.set_title('polyder1')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(fontsize='large', loc='upper right')
    ax2.invert_xaxis()
    plt.legend(loc=4)  # 指定legend的位置右下角
    savepath = 'pict'
    isExists = os.path.exists(savepath)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(savepath)
        print(savepath + ' 创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(savepath + ' 目录已存在')
    plt.savefig(savepath + '/' + name + '.png')
    plt.show()

def point(a,b,file):
    '''
    求出二阶导数的第一个零点作为拐点
    :param G:
    :param file:
    :return:
    '''
    df_y = pd.read_excel(file, usecols=[0])
    df_list = df_y.values.tolist()
    df_list = df_list[:math.ceil(len(df_list) / 2)]
    # 度数从大到小出现的次数
    cont_list = [a for b in df_list for a in b]
    df_x = [i for i in range(len(df_list))]
    x = df_x[:]
    y = cont_list[:]
    # 12次曲线拟合
    f12 = np.polyfit(x, y, 12)
    p12 = np.poly1d(f12)
    yvals = p12(x)  # 拟合曲线的y值
    p12_1 = np.polyder(p12, 1)  # 一阶导数
    p12_2 = np.polyder(p12, 2)  #二阶导数
    #biSection(a,b,1e-10,p12_2,p12)
    solution = root(fun=p12_2,x0=[10])
    print(solution.x[0])
    print(p12(solution.x[0]))



def biSection(a,b,threshold,f):
    '''
    二分法求函数零点
    :param a:
    :param b:
    :param threshold:
    :param f:
    :return:
    '''
    iter=0
    while a:
        mid = a + abs(b-a)/2.0
        if abs(f(mid)) < threshold:
            return mid
        if f(mid)*f(b) < 0:
            a = mid
        if f(a)*f(mid) < 0:
            b=mid
        iter+=1
        print(str(iter)+ " a= "+str(a)+ ", b= "+str(b)+', F(a)= '+str(f(a))+', F(b)= '+str(f(b)))


def func(G,name,file):
    '''
    使用截断机制,统计度以及其出现的频次
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
    #cont_list = [i for i in cont_list]
    df_x = pd.read_excel(file,usecols=[0])
    df_list = df_x.values.tolist()
    degree_list = [a for b in df_list for a in b]
    print(degree_list)
    x = degree_list[:]
    y = cont_list[:]
    polyfit(x,y,name)

def polyfit(x,y,name):
    '''
    使用曲线拟合
    :param x:
    :param y:
    :return:
    '''
    f10 = np.polyfit(x, y, 10)
    # 9次曲线拟合
    f9 = np.polyfit(x, y, 9)
    # 8次曲线拟合
    f8 = np.polyfit(x, y, 8)
    # 7次曲线拟合
    f7 = np.polyfit(x, y, 7)
    # 6次曲线拟合
    f6 = np.polyfit(x, y, 6)
    f5 = np.polyfit(x, y, 5)
    # print('f1 is:', f1)
    # 得到拟合的多项式，按照阶数从高到低
    # p10 = np.poly1d(f10)
    p9 = np.poly1d(f9)
    p8 = np.poly1d(f8)
    p7 = np.poly1d(f7)
    p6 = np.poly1d(f6)
    p5 = np.poly1d(f5)
    # print('p1 is :\n', p1)
    yvals = p9(x)  # 拟合曲线的y值
    # p10_1 = np.polyder(p10, 1)
    p9_1 = np.polyder(p9, 1)  # 一阶导数
    p8_1 = np.polyder(p8, 1)
    p7_1 = np.polyder(p7, 1)
    p6_1 = np.polyder(p6, 1)
    p5_1 = np.polyder(p5, 1)
    # print('p2 is:\n', p2)
    p9_2 = np.polyder(p9_1, 1)
    print('零点:', p9_2.r)
    y3 = p9_2(x)
    #yvals10 = p10_1(x)
    yvals9 = p9_1(x)
    yvals8 = p8_1(x)
    yvals7 = p7_1(x)
    yvals6 = p6_1(x)
    yvals5 = p5_1(x)
    # yvals2 = p3(x)

    # print('yvals is :\n', yvals)
    # 绘图
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(131)
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    # plt.gca().invert_xaxis()
    ax1.set_title('polyfitting of ' + name)
    ax1.set_xlabel('degree')
    ax1.set_ylabel('frequency')
    ax1.legend(fontsize='large', loc='upper left')
    ax2 = plt.subplot(132)
    # plot10 = plt.plot(x,yvals10,'m',label='polyder10 values')
    plot9 = plt.plot(x, yvals9, 'y', label='polyder9_1 values')
    plot8 = plt.plot(x, yvals8, 'r', label='polyder8_1 values')
    plot7 = plt.plot(x, yvals7, 'b', label='polyder7_1 values')
    # plot6 = plt.plot(x, yvals6, 'g', label='polyder6_1 values')
    # plot5 = plt.plot(x, yvals5, 'c', label='polyder5_1 values')
    # plt.gca().invert_xaxis()
    # plot3 = plt.plot(x, yvals2,'y',label = 'polyder2 values')
    ax2.set_title('polyder1')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(fontsize='large', loc='upper right')
    ax3 = plt.subplot(133)
    plot8 = plt.plot(x, y3, 'r', label='polyder9_2 values')
    # plt.gca().invert_xaxis()
    ax3.set_title('polyder2')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend(fontsize='large', loc='upper right')
    # my_x_ticks = np.arange(0, max, 80)
    # plt.xticks(my_x_ticks)
    plt.legend(loc=4)  # 指定legend的位置右下角
    p2 = p9_1
    # solution2 = root(fun=p3, x0=[1,2,3])
    solution = root(fun=p9_1, x0=[1])
    print(solution)
    # list_2 = list[i for i in list(p3(solution.x))]
    list_2 = []
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
    # return p1(solution2.x[0])


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

def part_degree(G,threshold):
    '''
    根据threhold将节点以及度列表进行切片,返回ID，degree，敏感度
    :param G:
    :param threshold:
    :return:
    '''
    values = []
    #从大到小
    threshold.reverse()
    dict1 = dict(sorted(nx.degree(G), key=lambda x: x[1], reverse=True))
    key = list(dict1.keys())
    value = list(dict1.values())
    num = 0
    before = max(value)
    for i in threshold:
        value_tmp = [i for i in filter(lambda x: before >= x > i,value)]
        values.append(value_tmp)
        before = i
    before = 0
    length = 0
    keys = []
    df = []
    for i in values:
        length += len(i)
        key_tmp = key[before:length]
        keys.append(key_tmp)
        value1 = i[:]
        value2 = value1[:]
        value2.pop(0)
        value1.pop()
        dis = list(map(lambda x:x[0]-x[1],zip(value1,value2)))
        dis_max = max(dis)
        df.append(math.ceil(dis_max/2))
        before = length
    return keys,values,df

def Laplace(G,threshold,epsilon,name):
    '''
    对value加入Laplace噪声
    :param G: 
    :param threshold: 
    :return: 
    '''
    key,value,df = part_degree(G,threshold)
    keys = [a for b in key for a in b]
    values = [a for b in value for a in b]
    high_degree = value.pop(0)
    high_keys = key.pop(0)
    high_df = df.pop(0)
    high_mean = np.mean(high_degree)
    high_noise = [high_mean for i in high_degree]
    L = high_df / 2
    noise = list(np.random.laplace(0, L, len(high_degree)))
    high_noise = list(map(lambda x: x[0] + x[1], zip(high_noise, noise)))
    high_noise = [round(i) for i in high_noise]
    degree_dis(high_degree,high_noise,name,'high',2)
    noise_degree = []
    noise_degree.extend(high_noise)
    for i in range(len(df)):
        L = df[i] / epsilon
        value_noise = value[i][:]
        noise = list(np.random.laplace(0, L, len(value_noise)))
        value_noise = list(map(lambda x: x[0] + x[1], zip(value_noise, noise)))
        value_noise = [round(i) for i in value_noise]
        noise_degree.extend(value_noise)
        degree_dis(value[i][:],value_noise,name,i,epsilon)
    savepath = 'pict' + '/' + name + '/'
    with open(savepath+name+'.pk','wb') as file_to_write:
        pickle.dump(noise_degree,file_to_write)


def degree_dis(dis1,dis2,name,i,epsilon):
    '''
    绘制加噪前后度分布的曲线，以及均值的变化
    :param dis1:
    :param dis2:
    :return:
    '''
    x = list(range(len(dis1)))
    y1 = dis1
    y2 = dis2
    plt.figure(figsize=(20, 10))
    plot2 = plt.plot(x, y2, 'r', label='DP distribution')
    plot1 = plt.plot(x, y1, 'b', label='original distribution')
    original_avg = np.mean(y1)
    DP_avg = np.mean(y2)
    plt.xlabel('x')
    plt.ylabel('degree')
    #plt.legend(loc='upper right',fontsize='large')  # 指定legend的位置右下角
    plt.legend(['DP_avg = ' + str(DP_avg),'original_avg = ' + str(original_avg)], loc='lower left')
    savepath = 'pict' + '/' + name + '/'
    isExists = os.path.exists(savepath)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(savepath)
        print(savepath + ' 创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(savepath + ' 目录已存在')
    plt.savefig(savepath + name + '-' + str(epsilon) + '-' +str(i) + '.png')

def Chung_Lu(G,file,name):
    '''
    将加完噪的序列转化为Chung-Lu图
    :param file:
    :return:
    '''
    with open(file, 'rb') as file_to_read:
        noise_degree = pickle.load(file_to_read)
    G_dict = dict(sorted(nx.degree(G), key=lambda x: x[1], reverse=True))
    G_key = list(G_dict.keys())
    G_degree = list(G_dict.values())
    length = len(noise_degree)
    tmp = G_degree[length:]
    noise_degree.extend(tmp)
    W = sum(noise_degree)
    n = nx.number_of_nodes(G)
    a = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            a[i][j] = (noise_degree[i] * noise_degree[j]) / W
    a = a + a.T - np.diag(np.diag(a))
    savepath = 'pict' + '/' + name + '/'
    with open(savepath + name + '_p' + '.pk', 'wb') as file_to_write:
        pickle.dump(a, file_to_write)

def noise_row_sum(file):
    '''
    将Chung-Lu图每个节点对应的元素按行相加
    :param file:
    :return:
    '''
    with open(file, 'rb') as file_to_read:
        noise_matrix = pickle.load(file_to_read)
        print(noise_matrix)

def degree_maintain(G,file):
    '''
    度最大节点保持百分比
    :param G:
    :param file:
    :return:
    '''
    G_dict = dict(sorted(nx.degree(G), key=lambda x: x[1], reverse=True))
    G_key = list(G_dict.keys())
    G_degree = list(G_dict.values())
    with open(file,'rb') as file_to_read:
        noise_degree = pickle.load(file_to_read)
    print(G_key)
    print(G_degree)
    print(noise_degree)

def pict_degree_maintain():
    '''
    度最大节点保持百分比绘图
    :param G:
    :param file:
    :return:
    '''
    from matplotlib.ticker import FuncFormatter
    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'
    x = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    y_face_1 = [0.928571429,0.952380952,0.976190476,0.976190476,0.976190476
                ,0.976190476,0.976190476,0.976190476,0.976190476,1]
    y_Email_1 = [0.985294118,0.991176471,0.991176471,0.997058824,0.997058824
                ,0.997058824,0.997058824,0.997058824,0.997058824,0.997058824]
    y_cond_1 = [0.940092166,0.976958525,0.98156682,0.98156682,0.99078341
                ,0.995391705,0.995391705,0.995391705,0.995391705,0.995391705]
    y_dblp_1 = [0.963517306,0.982538198,0.988462738,0.991580917,0.992828188
                ,0.99407546,0.995946367,0.996881821,0.997505457,0.998129093]
    y_face_5 = [0.985221675,0.995073892,0.995073892,0.995073892,0.995073892
                ,0.995073892,0.995073892,0.995073892,0.995073892,0.995073892]
    y_Email_5 = [0.997668998,0.997668998,0.997668998,0.997668998,0.997668998
                ,0.997668998,0.997668998,1,1,1]
    y_cond_5 = [0.997762864,0.997762864,0.997762864,0.997762864,0.997762864
                ,0.997762864,0.997762864,0.997762864,0.997762864,0.997762864]
    y_dblp_5 = [0.999834929,0.999834929,0.999834929,0.999834929,0.999834929
                ,0.999834929,0.999834929,0.999834929,0.999834929,0.999834929]
    plt.figure(figsize=(8, 8))
    plot1 = plt.plot(x, y_face_1, 'b', label='facebook')
    plot2 = plt.plot(x, y_Email_1, 'g', label='Email-Enron')
    plot3 = plt.plot(x, y_cond_1, 'r', label='CA-condMat')
    plot4 = plt.plot(x, y_dblp_1, 'y', label='dblp')
    plt.xlabel('epsilon')
    plt.ylabel('percentage')
    plt.title('1% similarity')
    plt.legend()
    #my_y_ticks = np.arange(0.98,1,0.002)
    #plt.yticks(my_y_ticks)
    #plt.ylim(0.98,1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()

def variance(G,threshold,epsilon,name):
    '''
    求出每个分组的标准差,根据标准差的激活值的大小,activation>0.95则需要进行处理
    :return:
    '''
    import math
    key, value, df = part_degree(G, threshold)
    value_copy = value[:]
    high_degree = value_copy.pop(0)
    high_df = df.pop(0)
    high_mean = np.mean(high_degree)
    high_noise = [high_mean for i in high_degree]
    L = high_df / 2
    noise = list(np.random.laplace(0, L, len(high_degree)))
    high_noise = list(map(lambda x: x[0] + x[1], zip(high_noise, noise)))
    high_noise = [round(i) for i in high_noise]
    noise_degree = []
    noise_degree.append(high_noise)
    value_noise = value[:]
    print(value_noise)
    value_noise.pop(0)
    list_std = []
    for i in range(len(value_noise)):
        std = np.std(value_noise[i],ddof=1)
        list_std.append(std)
    std_max = np.max(list_std)
    for i in range(len(value_noise)):
        std = np.std(value_noise[i],ddof=1)
        activation = 1/(1+math.exp(-std))
        print('std:',std)
        print('activation:',activation)
        if activation > 0.95:
            num = epsilon_Relu(std_max / std, 10)
            print('num:',num)
            tmp_epsilon = epsilon * num
            print('epsilon:',tmp_epsilon)
            L = df[i] / tmp_epsilon
            noise = list(np.random.laplace(0, L, len(value_noise[i])))
            value_noise[i] = list(map(lambda x: x[0] + x[1], zip(value_noise[i], noise)))
            value_noise[i] = [round(j) for j in value_noise[i]]
        noise_degree.append(value_noise[i])
    savepath = 'pict' + '/'
    with open(savepath + name  + '_group.pk', 'wb') as file_to_write:
        pickle.dump(noise_degree, file_to_write)
    print(noise_degree)

def epsilon_Relu(num,max_value):
    '''
    对于标准差使用Relu函数判断epsilon的大小关系
    :param num:
    :param threshold:
    :return:
    '''
    a = 0
    threshold = 0
    if num < threshold:
        result = a * (num - threshold)
    elif num < max_value:
        result = np.sqrt(num)
    else:
        result = max_value
    return result
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
    #对度以及对应出现的次数进行拟合后，进行截断获得的二阶导数零点
    facebook_2_low = 136.97061775
    facebook_2_high = 365.82809708
    facebook_2_list = [926.72474642, 722.77660609, 557.38709937, 365.82809708, 206.96154335, 136.97061775]
    Email_2_low = 118.76636075
    Email_2_high = 890.19454368
    Email_2_list = [1284.1022358,1106.3684035,890.19454368,657.82585429,442.38269369,259.64012053,118.76636075]
    cond_2_low = 38.68998649
    cond_2_high = 133.43065935
    cond_2_list = [259.74800381,217.2668708,171.06083706,133.43065935,96.6841193,62.41025877,38.68998649]
    dblp_2_low = 36.40478825
    dblp_2_high = 213.78199945
    dblp_2_list = [313.7991089,261.51650795,213.78199945,164.17030151,116.28585535,71.89060541,36.40478825]
    #Laplace(G_face_connect,facebook_2,0.5,'face')
    #对度的值分布进行曲线拟合，获得二阶导数零点
    face_fre_2_list = [136.97061775,1.39124414e+02,1.44556869e+02,1.54340456e+02
                        ,1.63938463e+02,1.74479112e+02,1.85497373e+02
                       ,1.95817312e+02,2.14705827e+02,2.25365647e+02]
    Email_fre_2_list = [118.76636075,126.96598235,139.27290664,157.64662113
                        ,185.2763339,214.05093964,257.68695449
                        ,294.09662417,409.45897402,468.50080861]
    cond_fre_2_list = [38.68998649,41.40127082,46.1536108,52.43562554
                       ,60.17928542,69.4313085,74.66892891
                       ,86.31875535,95.4288922,115.30925353,133.43065935]
    dblp_fre_2_list = [36.40478825,4.27936997e+01,5.28347889e+01,6.65825318e+01
                        ,8.31533309e+01,1.01546545e+02,1.20666296e+02
                        ,1.38770070e+02,1.7000e+02,1.74729408e+02,213.78199945]
    #degree_maintain(G_face_connect,'pict/face_1/face_1.pk')
    variance(G_face_connect,face_fre_2_list,1,'face_noise')
    variance(G_Email_connect, Email_fre_2_list, 1,'Email_noise')
    variance(G_cond_connect, cond_fre_2_list, 1,'cond_noise')
    variance(G_dblp_connect, dblp_fre_2_list, 1,'dblp_noise')
