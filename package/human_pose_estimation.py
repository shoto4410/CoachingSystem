import cv2
import os
import glob
import mediapipe as mp
import pandas as pd
import numpy as np
# from natsort import natsorted
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def df_columns_rename(df):
    df.columns = ["nose_x","nose_y","nose_z",
              "left_eye_inner_x","left_eye_inner_y","left_eye_inner_z",
              "left_eye_x","left_eye_y","left_eye_z",
              "left_eye_outer_x","left_eye_outer_y","left_eye_outer_z",
             "right_eye_inner_x","right_eye_inner_y","right_eye_inner_z",
              "right_eye_x","right_eye_y","right_eye_z",
              "right_eye_outer_x","right_eye_outer_y","right_eye_outer_z",
             "left_ear_x","left_ear_y","left_ear_z",
              "right_ear_x","right_ear_y","right_ear_z",
              "mouth_left_x","mouth_left_y","mouth_left_z",
              "mouth_right_x","mouth_right_y","mouth_right_z",
             "left_shoulder_x","left_shoulder_y","left_shoulder_z",
              "right_shoulder_x","right_shoulder_y","right_shoulder_z",
             "left_elbow_x","left_elbow_y","left_elbow_z",
              "right_elbow_x","right_elbow_y","right_elbow_z",
              "left_wrist_x","left_wrist_y","left_wrist_z",
              "right_wrist_x","right_wrist_y","right_wrist_z",
              "left_pinky_x","left_pinky_y","left_pinky_z",
              "right_pinky_x","right_pinky_y","right_pinky_z",
              "left_index_x","left_index_y","left_index_z",
              "right_index_x","right_index_y","right_index_z",
              "left_thumb_x","left_thumb_y","left_thumb_z",
              "right_thumb_x","right_thumb_y","right_thumb_z",
              "left_hip_x","left_hip_y","left_hip_z",
              "right_hip_x","right_hip_y","right_hip_z",
              "left_knee_x","left_knee_y","left_knee_z",
              "right_knee_x","right_knee_y","right_knee_z",
              "left_ankle_x","left_ankle_y","left_ankle_z",
              "right_ankle_x","right_ankle_y","right_ankle_z",
              "left_heel_x","left_heel_y","left_heel_z",
              "right_heel_x","right_heel_y","right_heel_z",
              "left_foot_index_x","left_foot_index_y","left_foot_index_z",
              "right_foot_index_x","right_foot_index_y","right_foot_index_z"
             ]
    return df

    
#csvに座標出力
def holistic_csv(FILENAME):
    print("座標出力開始")
    # dir = "data/time_crop/edited_output.mp4"
    # dir = 'data/movie/'+FILENAME+'.mp4'
    dir = 'data/edit_movie/'+FILENAME+'.mp4'
    
    data_land = np.zeros((0,99))
    # stream mp4 file
    cap = cv2.VideoCapture(dir)#load mp4 file
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            #get coordinate
            data_land2 = np.zeros((1,3))
            for x in range (0,33):
                #if results.pose_landmarks
                #print(results.pose_landmarks)
                if results.pose_landmarks == None:
                    data1 = None
                    data2 = None
                    data3 = None
                else:
                    data1 = results.pose_landmarks.landmark[x].x
                    data2 = results.pose_landmarks.landmark[x].y
                    data3 = results.pose_landmarks.landmark[x].z
                keydata = np.hstack((data1,data2,data3)).reshape(1,-1)
                data_land2 = np.hstack((data_land2,keydata))
            data_land2 = data_land2[:,3:]
            data_land = np.vstack((data_land,data_land2))
            # cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    #np.savetxt('keypoint_results.csv',data_land,delimiter = ',')
    #rint(data_land)
    df = pd.DataFrame(data_land)
    df = df_columns_rename(df)
    #df.to_excel("data/temp/result_csv/to_excel_out.xlsx", encoding = "utf-8")
    df.to_excel('csv/user/'+FILENAME+'.xlsx', encoding='utf-8')
    print("座標出力終了")

# holistic_csv("data/time_crop/edited_output.mp4")
# holistic_csv('user')

