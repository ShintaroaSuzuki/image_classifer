import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import glob
import os

num_generate_image = 1

dirs = glob.glob('data/_train/*')
names = []
for dir in dirs:
    dir = dir.split('\\')[-1]
    names.append(dir)

for name, dir in zip(names, dirs):
    save_path = dir  # 保存ディレクトリのパス
    img_dirs = glob.glob(save_path + '/*')

    for img_dir in img_dirs:
        img = image.load_img(img_dir)
        img = np.array(img)

        # 画像データ生成器を作成する。
        # -20° ~ 20° の範囲でランダムに回転を行う。
        datagen = image.ImageDataGenerator(rotation_range=20)

        x = img[np.newaxis]  #  (Height, Width, Channels)  -> (1, Height, Width, Channels)
        gen = datagen.flow(x, batch_size=1, save_to_dir=save_path,
                           save_prefix=name + '_cutted', save_format='jpg')

        for i in range(num_generate_image):
            # ミニバッチを生成したタイミングでディレクトリに
            # 画像が保存される。
            next(gen)
else:
    print('conplete image generator')
