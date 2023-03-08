from ast import arg
from cmath import nan
import pandas as pd
import openpyxl
import math
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import mediapipe as mp

from package.function import (cal_error, correctional_list, mint_coordinate_error, nine_to_three,
                      scaling, rotation, plot_landmarks, translation,
                      Correct_anomalies)
from package.human_pose_estimation import holistic_csv
from package.editvideo import segmantation


def coaching_system():
    FILENAME_USER = sys.argv[1]
    FILENAME_COPYCAT = sys.argv[2]
    THETA = float(sys.argv[3])
    X1, Y1, X2, Y2 = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])

    segmantation(FILENAME_USER, X1, Y1, X2, Y2)
    holistic_csv(FILENAME_USER)
    # df_user = pd.read_excel('csv/copycat/copycat.xlsx', index_col=0).values
    df_user = pd.read_excel('csv/user/'+FILENAME_USER+'.xlsx', index_col=0).values
    df_copycat = pd.read_excel('csv/copycat/'+FILENAME_COPYCAT+'.xlsx', index_col=0).values

    #numpy配列に変更
    C_user = nine_to_three(df_user)
    C_copycat = nine_to_three(df_copycat)

    #異常値検出・修正
    for r in range(10,5-1,-1):
        C_user = Correct_anomalies(C_user,r)
        C_copycat = Correct_anomalies(C_copycat,r)
    
    # #①平行移動
    C_user = translation(C_user)
    C_copycat = translation(C_copycat)
    
    #③回転
    C_user = rotation(C_user)
    C_copycat = rotation(C_copycat)
    #②スケーリング
    C_user = scaling(C_user)
    C_copycat = scaling(C_copycat)
    C_user *= -1
    C_copycat *= -1
    
    k = mint_coordinate_error(C_user, C_copycat)
    error = cal_error(C_user, C_copycat, k)
    elist = correctional_list(error, THETA)
    #for t in range(len(C_copycat)):
    for t in range(0, len(C_copycat), 3):
    #for t in range(len(C_copycat)-5, len(C_copycat)):
        plot_landmarks(C_user, C_copycat, t, k, elist)


def main():
    print('Program starting...')

    coaching_system()

    print('Program ending...')


if __name__ == '__main__':
    main()

