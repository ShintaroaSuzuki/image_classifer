from __future__ import print_function
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.optimizers import Adam
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

# 学習用データ
train = './data/_train/data.txt'
# 検証用データ
test = './data/_test/data.txt'
# 学習訓練の試行回数
epochs = 300
# 1回の学習で何枚の画像を使うか
batch_size = 50
# 学習率、小さすぎると学習が進まないし、大きすぎても誤差が収束しなかったり発散したりしてダメとか。繊細
learning_rate = 1e-04
# LeakyReLUのパラメータ
alpha = 0.01

# 識別ラベルの数(今回は3つ)
NUM_CLASSES = 3
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 64

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

with open(train, mode='r') as f:
    # データを入れる配列
    train_image = []
    train_label = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # データを読み込んで28x28に縮小
        img = load_img(l[0], target_size=(IMAGE_SIZE,IMAGE_SIZE))
        img_array  = img_to_array(img)
        train_image.append(img_array)
        # ラベルを追加
        train_label.append(l[1])
    # 整形
    train_image = np.asarray(train_image)
    train_image = train_image.astype('float32')
    train_image = train_image / 255.0
    train_label = np.asarray(train_label)
    train_label = np_utils.to_categorical(train_label, NUM_CLASSES)

with open(test, mode='r') as f:
    # データを入れる配列
    test_image = []
    test_label = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # データを読み込んで28x28に縮小
        img = load_img(l[0], target_size=(IMAGE_SIZE,IMAGE_SIZE))
        img_array  = img_to_array(img)
        test_image.append(img_array)
        # ラベルを追加
        test_label.append(l[1])
    # 整形
    test_image = np.asarray(test_image)
    test_image = test_image.astype('float32')
    test_image = test_image / 255.0
    test_label = np.asarray(test_label)
    test_label = np_utils.to_categorical(test_label, NUM_CLASSES)

model = Sequential()

model.add(Conv2D(8,(2, 2),input_shape=(IMAGE_SIZE, IMAGE_SIZE,3)))
model.add(Activation('relu'))

model.add(Conv2D(16,(2, 2),input_shape=(IMAGE_SIZE, IMAGE_SIZE,3)))
model.add(Activation('relu'))

model.add(Conv2D(32,(2, 2),input_shape=(IMAGE_SIZE, IMAGE_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))

adam = Adam(lr=learning_rate)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(train_image, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[early_stopping],
          validation_data = (test_image, test_label))  # callbacks=[early_stopping],

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()

# tensorflowで丁寧に実装しなきゃだめだ
