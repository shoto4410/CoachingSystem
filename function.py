from distutils.log import error
from re import T
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt


def mint_coordinate_error(C_user, C_copycat):
    len_user = len(C_user)
    len_copycat = len(C_copycat)

    num = len_user - len_copycat + 1
    min = 10**8
    #誤差が最小の時間のずれ
    min_error_t = 0
    #手本者のフロベニウスノルムの合計
    #fn_copycat = sum_frobenius_norm(C_copycat, len_copycat, 0)
    fn_copycat = sum_frobenius_norm(C_copycat, len_copycat, 0)
    for t in range(num):
        #fn_user = sum_frobenius_norm(C_user, len_copycat, t)
        fn_user = sum_frobenius_norm(C_user, len_copycat, t)
        error = np.abs(fn_user-fn_copycat)
        if min >= error:
            min = error
            min_error_t = t

    print('min t: {}'.format(min_error_t))
    return min_error_t


#長さL分のフロベニウスノルムの合計を算出
def sum_frobenius_norm(C, L, t):#L is length of C_coypycat. t is 時間のずれ
    fn = 0
    #3次元座標での計測
    # for i in range(L):
    #    fn += np.sqrt(np.sum(np.square(C[i+t,:])))#1行のフロベニウスノルム
    #2次元座標での計測
    for i in range(L):
        x = 0
        y = 0
        #z = 0
        for j in range(33):
            x += np.square(C[i+t,j,0])
            y += np.square(C[i+t,j,1])
            #z += np.square(C[i+t,j,1])
        fn += (x + y)

    return np.sqrt(fn)

#座標誤差を算出
def cal_error(C_user, C_copycat, min_error_t):
    len_copycat = len(C_copycat)
    C_user = C_user[min_error_t:len_copycat+min_error_t]

    C_error = np.abs(C_user - C_copycat)
    error = np.zeros((len_copycat,33))
    '''
    print(C_error)
    for i in range(len_copycat):
        for j in range(33):
            error[i, j] = np.sqrt(np.sum(np.square(C_error[i, j, :])))
    '''
    #print(C_error)
    for i in range(len_copycat):
        for j in range(33):
            sum = np.sum(np.square(C_error[i,j,0] + C_error[i,j,1]))
            error[i, j] = np.sqrt(sum)

    # sns.heatmap(error, cmap='Greys')
    # plt.show()

    return error

#閾値よりも誤差が大きい部位を格納
def correctional_list(error, theta):
    str = []
    #theta = 0.5
    for i in range(error.shape[0]):
        for j in range(error.shape[1]):
            if error[i, j] > theta:
                str.append(j)
    str.sort()
    list = set(str)
    print(list)
    return list


#コサインの計算
def cal_cos_theta(left_shoulder, right_shoulder):
    x = np.array([1, 0, 0])
    #11.left_shoulder, 12.right_shoulder
    a = np.array([left_shoulder[0], left_shoulder[1], left_shoulder[2]])
    b = np.array([right_shoulder[0], right_shoulder[1], right_shoulder[2]])
    d = a - b
    cos_theta =  np.dot(d, x) / (np.linalg.norm(d) * np.linalg.norm(x))

    return cos_theta

def translation(C):
    #l = shoulder_average(C[0,:])
    #C = C - l
    '''
    for i in range(len(C)):
        l = shoulder_average(C[0,:])
        C[i,:,:] -= l
    '''
    for i in range(len(C)):#ここの長さが間違っていた
        ls = (C[i,11,:] + C[i,12,:])/2
        C[i,:,:] -= ls
    return C

def shoulder_average(C):
    #l = (C[11]-C[12])/2
    #l = (C[11]+C[12])/2
    l = (np.abs(C[11])+np.abs(C[12])) / 2
    #print(l)
    return l

#ｌは絶対値の必要があるのか検証する

#②スケーリング
def scaling(C):
    l1 = len(C)
    l2 = 33
    for t in range(l1):
        # l = (C[t,11,:]+C[t,12,:]+C[t,23,:]+C[t,24,:]) / 4
        # sx, sy, sz = l[0], l[1], l[2]
        ex = (math.sqrt((C[t,11,0]-C[t,24,0])**2) + math.sqrt((C[t,12,0]-C[t,23,0])**2)) / 2
        ey = (math.sqrt((C[t,11,1]-C[t,24,1])**2) + math.sqrt((C[t,12,1]-C[t,23,1])**2)) / 2
        ez = (math.sqrt((C[t,11,2]-C[t,24,2])**2) + math.sqrt((C[t,12,2]-C[t,23,2])**2)) / 2
        S = np.array([[1/ex, 0, 0],
                    [0, 1/ey, 0],
                    [0, 0, 1/ez]])
        for i in range(l2):
            #C[t,i,:] = np.dot(Ry, np.dot(Ry, C[t,i,:]))
            C[t,i,:] = np.dot(S, C[t,i,:])
    
    return C

#③回転
#毎フレーム回転させる
def rotation(C):
    for t in range(C.shape[0]):
        cos_theta = cal_cos_theta(C[t,11,:], C[t,12,:])
        sin_theta = math.sqrt(1-cos_theta**2)
        #print(cos_theta*180/np.pi, sin_theta*180/np.pi)

        Rx = np.array([[1, 0, 0],
                    [0, cos_theta, sin_theta],
                    [0, -sin_theta, cos_theta]])
        Ry = np.array([[cos_theta, 0, -sin_theta],
                    [0, 1, 0],
                    [sin_theta, 0, cos_theta]])
        Rz = np.array([[cos_theta, sin_theta, 0],
                    [-sin_theta, cos_theta, 0],
                    [0, 0, 1]])
        
        for i in range(C.shape[1]):
            C[t,i,:] = np.dot(Ry, C[t,i,:])
    return C

#(o,o,o,o,o,o,...,o)99->((o,o,o),(o,o,o),(o,o,o),(o,o,o),...,(o,o,o))33
def nine_to_three(C):
    t = C.shape[0]
    i = 33
    j = 3
    a = np.zeros((t,i,j))

    for t in range(t):
        for i in range(33):
            a[t, i, 0] = C[t, i*3]
            a[t, i, 1] = C[t, i*3+1]
            a[t, i, 2] = C[t, i*3+2]

    return a

def Correct_anomalies(C, theta):
    l1 = len(C)
    l2 = 33
    count = 0
    for i in range(1, l1-1):
        for j in range(l2):
            avex = (C[i-1,j,0]+C[i+1,j,0]) / 2
            avey = (C[i-1,j,1]+C[i+1,j,1]) / 2
            avez = (C[i-1,j,2]+C[i+1,j,2]) / 2

            if np.abs(C[i,j,0]-avex)*100 > theta:
                C[i,j,0] = avex
                count += 1
            if np.abs(C[i,j,1]-avey)*100 > theta:
                C[i,j,1] = avey
                count += 1
            if np.abs(C[i,j,2]-avez)*100 > theta:
                C[i,j,2] = avez
                count += 1
    #print('count:{}'.format(count))
    return C

#3->2
def three_to_two(C):
    t = C.shape[0]
    i = 33
    j = 2
    a = np.zeros((t,i,j))
    
    for t in range(t):
        for i in range(33):
            a[t,i,0] = C[t,i,0]
            a[t,i,1] = C[t,i,1]
            
    return a

def plot_landmarks(c1, c2, t, k, elist):
    x1 = np.zeros(33)
    y1 = np.zeros(33)
    #z1 = np.zeros(33)
    x2 = np.zeros(33)
    y2 = np.zeros(33)
    #z2 = np.zeros(33)

    for i in range(33):
        x1[i] = c1[t+k,i,0]
        y1[i] = c1[t+k,i,1]
        #z1[i] = c1[t+k,i,2]
        x2[i] = c2[t,i,0]
        y2[i] = c2[t,i,1]
        #z2[i] = c2[t,i,2]

    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    ax.scatter(x1, y1)
    ax.plot([x1[10],x1[9]],[y1[10],y1[9]],color='black')#口
    ax.plot([x1[1],x1[2]],[y1[1],y1[2]],color='black')#
    ax.plot([x1[2],x1[3]],[y1[2],y1[3]],color='black')#
    ax.plot([x1[3],x1[7]],[y1[3],y1[7]],color='black')#
    ax.plot([x1[4],x1[5]],[y1[4],y1[5]],color='black')#
    ax.plot([x1[5],x1[6]],[y1[5],y1[6]],color='black')#
    ax.plot([x1[6],x1[8]],[y1[6],y1[8]],color='black')#
    ax.plot([x1[0],x1[1]],[y1[0],y1[1]],color='black')#
    ax.plot([x1[0],x1[4]],[y1[0],y1[4]],color='black')#
    ax.plot([x1[11],x1[13]],[y1[11],y1[13]],color='black')#左肩肘
    ax.plot([x1[15],x1[13]],[y1[15],y1[13]],color='black')#
    ax.plot([x1[14],x1[12]],[y1[14],y1[12]],color='black')#右肩肘
    ax.plot([x1[14],x1[16]],[y1[14],y1[16]],color='black')
    ax.plot([x1[11],x1[12]],[y1[11],y1[12]],color='black')#肩
    ax.plot([x1[11],x1[23]],[y1[11],y1[23]],color='black')#肩腰
    ax.plot([x1[24],x1[12]],[y1[24],y1[12]],color='black')#肩腰
    ax.plot([x1[23],x1[24]],[y1[23],y1[24]],color='black')#腰
    ax.plot([x1[23],x1[25]],[y1[23],y1[25]],color='black')#左足膝
    ax.plot([x1[25],x1[27]],[y1[25],y1[27]],color='black')#
    ax.plot([x1[29],x1[27]],[y1[29],y1[27]],color='black')#
    ax.plot([x1[31],x1[27]],[y1[31],y1[27]],color='black')#
    ax.plot([x1[29],x1[31]],[y1[29],y1[31]],color='black')#
    ax.plot([x1[24],x1[26]],[y1[24],y1[26]],color='black')#右足膝
    ax.plot([x1[26],x1[28]],[y1[26],y1[28]],color='black')#
    ax.plot([x1[30],x1[28]],[y1[30],y1[28]],color='black')#
    ax.plot([x1[32],x1[28]],[y1[32],y1[28]],color='black')#
    ax.plot([x1[32],x1[30]],[y1[32],y1[30]],color='black')#

    if 11 in elist and 13 in elist:
        ax.plot([x1[11],x1[13]],[y1[11],y1[13]],color='red')#左肩肘
    if 13 in elist and 15 in elist:
        ax.plot([x1[15],x1[13]],[y1[15],y1[13]],color='red')#
    if 14 in elist and 12 in elist:
        ax.plot([x1[14],x1[12]],[y1[14],y1[12]],color='red')#右肩肘
    if 14 in elist and 16 in elist:
        ax.plot([x1[14],x1[16]],[y1[14],y1[16]],color='red')
    if 11 in elist and 12 in elist:
        ax.plot([x1[11],x1[12]],[y1[11],y1[12]],color='red')#肩
    if 11 in elist and 13 in elist:
        ax.plot([x1[11],x1[23]],[y1[11],y1[23]],color='red')#肩腰
    if 12 in elist and 24 in elist:
        ax.plot([x1[24],x1[12]],[y1[24],y1[12]],color='red')#肩腰
    # if 23 in elist and 24 in elist:
    #     ax.plot([x1[23],x1[24]],[y1[23],y1[24]],color='red')#腰
    ax.plot([x1[23],x1[25]],[y1[23],y1[25]],color='black')#左足膝
    ax.plot([x1[25],x1[27]],[y1[25],y1[27]],color='black')#
    ax.plot([x1[29],x1[27]],[y1[29],y1[27]],color='black')#
    ax.plot([x1[31],x1[27]],[y1[31],y1[27]],color='black')#
    ax.plot([x1[29],x1[31]],[y1[29],y1[31]],color='black')#

    ax.scatter(x2, y2)
    ax.plot([x2[10],x2[9]],[y2[10],y2[9]],color='blue')#口
    ax.plot([x2[1],x2[2]],[y2[1],y2[2]],color='blue')#
    ax.plot([x2[2],x2[3]],[y2[2],y2[3]],color='blue')#
    ax.plot([x2[3],x2[7]],[y2[3],y2[7]],color='blue')#
    ax.plot([x2[4],x2[5]],[y2[4],y2[5]],color='blue')#
    ax.plot([x2[5],x2[6]],[y2[5],y2[6]],color='blue')#
    ax.plot([x2[6],x2[8]],[y2[6],y2[8]],color='blue')#
    ax.plot([x2[0],x2[1]],[y2[0],y2[1]],color='blue')#
    ax.plot([x2[0],x2[4]],[y2[0],y2[4]],color='blue')#
    ax.plot([x2[11],x2[13]],[y2[11],y2[13]],color='blue')#左肩肘
    ax.plot([x2[15],x2[13]],[y2[15],y2[13]],color='blue')#
    ax.plot([x2[14],x2[12]],[y2[14],y2[12]],color='blue')#右肩肘
    ax.plot([x2[14],x2[16]],[y2[14],y2[16]],color='blue')
    ax.plot([x2[11],x2[12]],[y2[11],y2[12]],color='blue')#肩
    ax.plot([x2[11],x2[23]],[y2[11],y2[23]],color='blue')#肩腰
    ax.plot([x2[24],x2[12]],[y2[24],y2[12]],color='blue')#肩腰
    ax.plot([x2[23],x2[24]],[y2[23],y2[24]],color='blue')#腰
    ax.plot([x2[23],x2[25]],[y2[23],y2[25]],color='blue')#左足膝
    ax.plot([x2[25],x2[27]],[y2[25],y2[27]],color='blue')#
    ax.plot([x2[29],x2[27]],[y2[29],y2[27]],color='blue')#
    ax.plot([x2[31],x2[27]],[y2[31],y2[27]],color='blue')#
    ax.plot([x2[29],x2[31]],[y2[29],y2[31]],color='blue')#
    ax.plot([x2[24],x2[26]],[y2[24],y2[26]],color='blue')#右足膝
    ax.plot([x2[26],x2[28]],[y2[26],y2[28]],color='blue')#
    ax.plot([x2[30],x2[28]],[y2[30],y2[28]],color='blue')#
    ax.plot([x2[32],x2[28]],[y2[32],y2[28]],color='blue')#
    ax.plot([x2[32],x2[30]],[y2[32],y2[30]],color='blue')#

    #ax.plot([x[],x[]],[y[],y[]],[z[],z[]],color='')#
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def plot3D(c1, c2, t, k):
    x1 = np.zeros(33)
    y1 = np.zeros(33)
    z1 = np.zeros(33)
    x2 = np.zeros(33)
    y2 = np.zeros(33)
    z2 = np.zeros(33)

    for i in range(33):
        x1[i] = c1[t+k,i,0]
        y1[i] = c1[t+k,i,1]
        z1[i] = c1[t+k,i,2]
        x2[i] = c2[t,i,0]
        y2[i] = c2[t,i,1]
        z2[i] = c2[t,i,2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1, y1, z1)
    ax.plot([x1[10],x1[9]],[y1[10],y1[9]],[z1[10],z1[9]],color='black')#口
    ax.plot([x1[1],x1[2]],[y1[1],y1[2]],[z1[1],z1[2]],color='black')#
    ax.plot([x1[2],x1[3]],[y1[2],y1[3]],[z1[2],z1[3]],color='black')#
    ax.plot([x1[3],x1[7]],[y1[3],y1[7]],[z1[3],z1[7]],color='black')#
    ax.plot([x1[4],x1[5]],[y1[4],y1[5]],[z1[4],z1[5]],color='black')#
    ax.plot([x1[5],x1[6]],[y1[5],y1[6]],[z1[5],z1[6]],color='black')#
    ax.plot([x1[6],x1[8]],[y1[6],y1[8]],[z1[6],z1[8]],color='black')#
    ax.plot([x1[0],x1[1]],[y1[0],y1[1]],[z1[0],z1[1]],color='black')#
    ax.plot([x1[0],x1[4]],[y1[0],y1[4]],[z1[0],z1[4]],color='black')#
    ax.plot([x1[11],x1[13]],[y1[11],y1[13]],[z1[11],z1[13]],color='black')#左肩肘
    ax.plot([x1[15],x1[13]],[y1[15],y1[13]],[z1[15],z1[13]],color='black')#
    ax.plot([x1[14],x1[12]],[y1[14],y1[12]],[z1[14],z1[12]],color='black')#右肩肘
    ax.plot([x1[14],x1[16]],[y1[14],y1[16]],[z1[14],z1[16]],color='black')
    ax.plot([x1[11],x1[12]],[y1[11],y1[12]],[z1[11],z1[12]],color='black')#肩
    ax.plot([x1[11],x1[23]],[y1[11],y1[23]],[z1[11],z1[23]],color='black')#肩腰
    ax.plot([x1[24],x1[12]],[y1[24],y1[12]],[z1[24],z1[12]],color='black')#肩腰
    ax.plot([x1[23],x1[24]],[y1[23],y1[24]],[z1[23],z1[24]],color='black')#腰
    ax.plot([x1[23],x1[25]],[y1[23],y1[25]],[z1[23],z1[25]],color='black')#左足膝
    ax.plot([x1[25],x1[27]],[y1[25],y1[27]],[z1[25],z1[27]],color='black')#
    ax.plot([x1[29],x1[27]],[y1[29],y1[27]],[z1[29],z1[27]],color='black')#
    ax.plot([x1[31],x1[27]],[y1[31],y1[27]],[z1[31],z1[27]],color='black')#
    ax.plot([x1[29],x1[31]],[y1[29],y1[31]],[z1[29],z1[31]],color='black')#
    ax.plot([x1[24],x1[26]],[y1[24],y1[26]],[z1[24],z1[26]],color='black')#右足膝
    ax.plot([x1[26],x1[28]],[y1[26],y1[28]],[z1[26],z1[28]],color='black')#
    ax.plot([x1[30],x1[28]],[y1[30],y1[28]],[z1[30],z1[28]],color='black')#
    ax.plot([x1[32],x1[28]],[y1[32],y1[28]],[z1[32],z1[28]],color='black')#
    ax.plot([x1[32],x1[30]],[y1[32],y1[30]],[z1[32],z1[30]],color='black')#

    ax.scatter(x2, y2, z2)
    ax.plot([x2[10],x2[9]],[y2[10],y2[9]],[z2[10],z2[9]],color='blue')#口
    ax.plot([x2[1],x2[2]],[y2[1],y2[2]],[z2[1],z2[2]],color='blue')#
    ax.plot([x2[2],x2[3]],[y2[2],y2[3]],[z2[2],z2[3]],color='blue')#
    ax.plot([x2[3],x2[7]],[y2[3],y2[7]],[z2[3],z2[7]],color='blue')#
    ax.plot([x2[4],x2[5]],[y2[4],y2[5]],[z2[4],z2[5]],color='blue')#
    ax.plot([x2[5],x2[6]],[y2[5],y2[6]],[z2[5],z2[6]],color='blue')#
    ax.plot([x2[6],x2[8]],[y2[6],y2[8]],[z2[6],z2[8]],color='blue')#
    ax.plot([x2[0],x2[1]],[y2[0],y2[1]],[z2[0],z2[1]],color='blue')#
    ax.plot([x2[0],x2[4]],[y2[0],y2[4]],[z2[0],z2[4]],color='blue')#
    ax.plot([x2[11],x2[13]],[y2[11],y2[13]],[z2[11],z2[13]],color='blue')#左肩肘
    ax.plot([x2[15],x2[13]],[y2[15],y2[13]],[z2[15],z2[13]],color='blue')#
    ax.plot([x2[14],x2[12]],[y2[14],y2[12]],[z2[14],z2[12]],color='blue')#右肩肘
    ax.plot([x2[14],x2[16]],[y2[14],y2[16]],[z2[14],z2[16]],color='blue')
    ax.plot([x2[11],x2[12]],[y2[11],y2[12]],[z2[11],z2[12]],color='blue')#肩
    ax.plot([x2[11],x2[23]],[y2[11],y2[23]],[z2[11],z2[23]],color='blue')#肩腰
    ax.plot([x2[24],x2[12]],[y2[24],y2[12]],[z2[24],z2[12]],color='blue')#肩腰
    ax.plot([x2[23],x2[24]],[y2[23],y2[24]],[z2[23],z2[24]],color='blue')#腰
    ax.plot([x2[23],x2[25]],[y2[23],y2[25]],[z2[23],z2[25]],color='blue')#左足膝
    ax.plot([x2[25],x2[27]],[y2[25],y2[27]],[z2[25],z2[27]],color='blue')#
    ax.plot([x2[29],x2[27]],[y2[29],y2[27]],[z2[29],z2[27]],color='blue')#
    ax.plot([x2[31],x2[27]],[y2[31],y2[27]],[z2[31],z2[27]],color='blue')#
    ax.plot([x2[29],x2[31]],[y2[29],y2[31]],[z2[29],z2[31]],color='blue')#
    ax.plot([x2[24],x2[26]],[y2[24],y2[26]],[z2[24],z2[26]],color='blue')#右足膝
    ax.plot([x2[26],x2[28]],[y2[26],y2[28]],[z2[26],z2[28]],color='blue')#
    ax.plot([x2[30],x2[28]],[y2[30],y2[28]],[z2[30],z2[28]],color='blue')#
    ax.plot([x2[32],x2[28]],[y2[32],y2[28]],[z2[32],z2[28]],color='blue')#
    ax.plot([x2[32],x2[30]],[y2[32],y2[30]],[z2[32],z2[30]],color='blue')#

    #ax.plot([x[],x[]],[y[],y[]],[z[],z[]],color='')#
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot2(c1, c2, t, k, elist):
    x1 = np.zeros(33)
    y1 = np.zeros(33)
    z1 = np.zeros(33)
    x2 = np.zeros(33)
    y2 = np.zeros(33)
    z2 = np.zeros(33)

    for i in range(33):
        x1[i] = c1[t+k,i,0]
        y1[i] = c1[t+k,i,1]
        z1[i] = c1[t+k,i,2]
        x2[i] = c2[t,i,0]
        y2[i] = c2[t,i,1]
        z2[i] = c2[t,i,2]

    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    ax.scatter(x1, y1)
    ax.plot([x1[10],x1[9]],[y1[10],y1[9]],color='black')#口
    ax.plot([x1[1],x1[2]],[y1[1],y1[2]],color='black')#
    ax.plot([x1[2],x1[3]],[y1[2],y1[3]],color='black')#
    ax.plot([x1[3],x1[7]],[y1[3],y1[7]],color='black')#
    ax.plot([x1[4],x1[5]],[y1[4],y1[5]],color='black')#
    ax.plot([x1[5],x1[6]],[y1[5],y1[6]],color='black')#
    ax.plot([x1[6],x1[8]],[y1[6],y1[8]],color='black')#
    ax.plot([x1[0],x1[1]],[y1[0],y1[1]],color='black')#
    ax.plot([x1[0],x1[4]],[y1[0],y1[4]],color='black')#
    ax.plot([x1[11],x1[13]],[y1[11],y1[13]],color='black')#左肩肘
    ax.plot([x1[15],x1[13]],[y1[15],y1[13]],color='black')#
    ax.plot([x1[14],x1[12]],[y1[14],y1[12]],color='black')#右肩肘
    ax.plot([x1[14],x1[16]],[y1[14],y1[16]],color='black')
    ax.plot([x1[11],x1[12]],[y1[11],y1[12]],color='black')#肩
    ax.plot([x1[11],x1[23]],[y1[11],y1[23]],color='black')#肩腰
    ax.plot([x1[24],x1[12]],[y1[24],y1[12]],color='black')#肩腰
    ax.plot([x1[23],x1[24]],[y1[23],y1[24]],color='black')#腰
    ax.plot([x1[23],x1[25]],[y1[23],y1[25]],color='black')#左足膝
    ax.plot([x1[25],x1[27]],[y1[25],y1[27]],color='black')#
    ax.plot([x1[29],x1[27]],[y1[29],y1[27]],color='black')#
    ax.plot([x1[31],x1[27]],[y1[31],y1[27]],color='black')#
    ax.plot([x1[29],x1[31]],[y1[29],y1[31]],color='black')#
    ax.plot([x1[24],x1[26]],[y1[24],y1[26]],color='black')#右足膝
    ax.plot([x1[26],x1[28]],[y1[26],y1[28]],color='black')#
    ax.plot([x1[30],x1[28]],[y1[30],y1[28]],color='black')#
    ax.plot([x1[32],x1[28]],[y1[32],y1[28]],color='black')#
    ax.plot([x1[32],x1[30]],[y1[32],y1[30]],color='black')#
    
    if 11 in elist and 13 in elist:
        ax.plot([x1[11],x1[13]],[y1[11],y1[13]],color='red')#左肩肘
    if 13 in elist and 15 in elist:
        ax.plot([x1[15],x1[13]],[y1[15],y1[13]],color='red')#
    if 14 in elist and 12 in elist:
        ax.plot([x1[14],x1[12]],[y1[14],y1[12]],color='red')#右肩肘
    if 14 in elist and 16 in elist:
        ax.plot([x1[14],x1[16]],[y1[14],y1[16]],color='red')
    if 11 in elist and 12 in elist:
        ax.plot([x1[11],x1[12]],[y1[11],y1[12]],color='red')#肩
    if 11 in elist and 13 in elist:
        ax.plot([x1[11],x1[23]],[y1[11],y1[23]],color='red')#肩腰
    if 12 in elist and 24 in elist:
        ax.plot([x1[24],x1[12]],[y1[24],y1[12]],color='red')#肩腰
    # if 23 in elist and 24 in elist:
    #     ax.plot([x1[23],x1[24]],[y1[23],y1[24]],color='red')#腰
    ax.plot([x1[23],x1[25]],[y1[23],y1[25]],color='black')#左足膝
    ax.plot([x1[25],x1[27]],[y1[25],y1[27]],color='black')#
    ax.plot([x1[29],x1[27]],[y1[29],y1[27]],color='black')#
    ax.plot([x1[31],x1[27]],[y1[31],y1[27]],color='black')#
    ax.plot([x1[29],x1[31]],[y1[29],y1[31]],color='black')#

    ax.scatter(x2, y2)
    ax.plot([x2[10],x2[9]],[y2[10],y2[9]],color='blue')#口
    ax.plot([x2[1],x2[2]],[y2[1],y2[2]],color='blue')#
    ax.plot([x2[2],x2[3]],[y2[2],y2[3]],color='blue')#
    ax.plot([x2[3],x2[7]],[y2[3],y2[7]],color='blue')#
    ax.plot([x2[4],x2[5]],[y2[4],y2[5]],color='blue')#
    ax.plot([x2[5],x2[6]],[y2[5],y2[6]],color='blue')#
    ax.plot([x2[6],x2[8]],[y2[6],y2[8]],color='blue')#
    ax.plot([x2[0],x2[1]],[y2[0],y2[1]],color='blue')#
    ax.plot([x2[0],x2[4]],[y2[0],y2[4]],color='blue')#
    ax.plot([x2[11],x2[13]],[y2[11],y2[13]],color='blue')#左肩肘
    ax.plot([x2[15],x2[13]],[y2[15],y2[13]],color='blue')#
    ax.plot([x2[14],x2[12]],[y2[14],y2[12]],color='blue')#右肩肘
    ax.plot([x2[14],x2[16]],[y2[14],y2[16]],color='blue')
    ax.plot([x2[11],x2[12]],[y2[11],y2[12]],color='blue')#肩
    ax.plot([x2[11],x2[23]],[y2[11],y2[23]],color='blue')#肩腰
    ax.plot([x2[24],x2[12]],[y2[24],y2[12]],color='blue')#肩腰
    ax.plot([x2[23],x2[24]],[y2[23],y2[24]],color='blue')#腰
    ax.plot([x2[23],x2[25]],[y2[23],y2[25]],color='blue')#左足膝
    ax.plot([x2[25],x2[27]],[y2[25],y2[27]],color='blue')#
    ax.plot([x2[29],x2[27]],[y2[29],y2[27]],color='blue')#
    ax.plot([x2[31],x2[27]],[y2[31],y2[27]],color='blue')#
    ax.plot([x2[29],x2[31]],[y2[29],y2[31]],color='blue')#
    ax.plot([x2[24],x2[26]],[y2[24],y2[26]],color='blue')#右足膝
    ax.plot([x2[26],x2[28]],[y2[26],y2[28]],color='blue')#
    ax.plot([x2[30],x2[28]],[y2[30],y2[28]],color='blue')#
    ax.plot([x2[32],x2[28]],[y2[32],y2[28]],color='blue')#
    ax.plot([x2[32],x2[30]],[y2[32],y2[30]],color='blue')#

    #ax.plot([x[],x[]],[y[],y[]],[z[],z[]],color='')#
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()