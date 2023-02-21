from moviepy.editor import *
import glob
from natsort import natsorted
import cv2
import os

#撮影時間の切り取り
def cut_movie(FILENAME, start, end):
    clip1 = VideoFileClip(FILENAME).subclip(start, end)

    final_clip = concatenate_videoclips([clip1])
    final_clip.write_videofile('data/time_crop/'+FILENAME+'.mp4')

#撮影範囲の切り取り
def crop_movie(FILENAME, X1, Y1, X2, Y2):
    file_path = "data/movie/"+FILENAME+".mp4"#トリミングしたい動画のパス

    X1= X1

    Y1 = Y1

    X2 = X2

    Y2 = Y2

    save_path = "data/movie/"+FILENAME+"_crop.mp4"#保存先のパス

    video = (VideoFileClip(file_path).crop(x1=X1,y1=Y1,x2=X2,y2=Y2))#トリミング実行

    video.write_videofile(save_path,fps=29) #保存

def segmantation(FILENAME, X1, Y1, X2, Y2):
    print('画像処理中...')
    IMG_PATH = 'data/images/'
    save_all_frames('data/movie/'+FILENAME+'.mp4', IMG_PATH, 'video_img', 'jpg')

    create_movie(FILENAME, X1, Y1, X2, Y2)
    # imgファイルを削除　→ 後の比較の際に混ざらないようにするため
    # for f in os.listdir(IMG_PATH):
    #     os.remove(os.path.join(IMG_PATH, f))
    
    print('画像処理終了')

#動画(mp4)から静止画(jpg)に変換
def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return


#初めのフレームを変更後、静止画(jpg)から動画(mp4)に変換
def create_movie(FILENAME, X1, Y1, X2, Y2):
    width, height = 1280,  720
    size=(width, height)#サイズ指定
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')#保存形式
    save = cv2.VideoWriter('data/edit_movie/'+FILENAME+'.mp4',fourcc,30.0,size)#動画を保存する形を作成

    img_list = glob.glob("data/images/*.jpg")
    img_list = natsorted(img_list)

    for i in range(len(img_list)):
        img = img_list[i]
        img = cv2.imread(img)

        if i == 0:
            # cv2.rectangle(img=img, pt1=(X1, Y1), pt2=(X2, Y2), color=(0,0,0), thickness=-1)
            cv2.rectangle(img=img, pt1=(0,0),  pt2=(width, Y1), color=(0,0,0), thickness=-1)
            cv2.rectangle(img=img, pt1=(0,Y1),  pt2=(X1,Y2), color=(0,0,0), thickness=-1)
            cv2.rectangle(img=img, pt1=(X2,Y1),  pt2=(width, Y2), color=(0,0,0), thickness=-1)
            cv2.rectangle(img=img, pt1=(0,Y2),  pt2=(width, height), color=(0,0,0), thickness=-1)

        save.write(img)

    save.release()
# image size is (1280, 720)
# segmantation('user', 600, 0, 1100, 650)

# img = cv2.imread('data/images/video_img_000.jpg')
# print(img.size)
# print(720*1280)

# X1, Y1, X2, Y2 = 600, 0, 1100, 650
# width, height = 1280, 720
# for i in range(width):
#     for j in range(height):
#         if i > X1 and i < X2 and j > Y1 and j < Y2:
#             a = 0