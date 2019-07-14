import cv2
import numpy as np
import glob
import os

def cut_face():
    # ディレクトリリスト
    dirs = glob.glob("./data/*")

    # OpenCVのデフォルトの分類器のpath。(https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xmlのファイルを使う)
    cascade_path = 'C:/Users/shinb/AppData/Local/Continuum/anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascade_path)

    for dir in dirs:
        # 先ほど集めてきた画像データのあるディレクトリ
        input_data_path = dir
        # 切り抜いた画像の保存先ディレクトリ(予めディレクトリを作っておいてください)
        save_path = dir + '_cutted'
        os.makedirs(save_path, exist_ok=True)

        # 収集した画像の枚数(任意で変更)
        image_count = len(os.listdir(dir))
        # 顔検知に成功した数(デフォルトで0を指定)
        face_detect_count = 0

        # 集めた画像データから顔が検知されたら、切り取り、保存する。
        for i in range(image_count):
            img = cv2.imread(input_data_path + '/' + input_data_path.split('\\')[-1] + '_' + str(i) + '.jpg', cv2.IMREAD_COLOR)
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                continue
            face = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100,100))
            if len(face) > 0:
                for rect in face:
                    # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
                    # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
                    # cv2.imwrite('detected.jpg', img)
                    x = rect[0]
                    y = rect[1]
                    w = rect[2]
                    h = rect[3]
                    cv2.imwrite(save_path + '/' + input_data_path.split('\\')[-1] + '_cutted' + '_' + str(face_detect_count) + '.jpg', img[y:y+h, x:x+w])
                    face_detect_count = face_detect_count + 1
            else:
                print(input_data_path.split('\\')[-1] + '_'  + str(i) + ':NoFace')
    print('顔の切り抜き完了')

if __name__ == '__main__':
    cut_face()
