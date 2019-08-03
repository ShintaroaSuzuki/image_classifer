import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img

cascade_path = 'C:/Users/shinb/AppData/Local/Continuum/anaconda3/Library/etc/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

img = cv2.imread('testimage.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100,100))
if len(face) > 0:
    for i, rect in enumerate(face):
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        cv2.imwrite('testimage' + str(i) + '.jpg', img[y:y+h, x:x+w])

IMAGE_SIZE = 64

img = load_img('testimage0.jpg', target_size=(IMAGE_SIZE,IMAGE_SIZE))
img_array  = img_to_array(img)

# 整形
img_array = np.asarray([img_array])
img_array = img_array.astype('float32')
img_array = img_array / 255.0

# TensorFlowのセッション
sess = tf.Session()

# 訓練済みモデルのmetaファイルを読み込み
saver = tf.train.import_meta_graph('model.ckpt.meta')

saver.restore(sess, 'model.ckpt')

res = sess.run('y_:0', feed_dict={
    'x:0': img_array,
    'is_training:0': False
})


"""
# モデルの復元
saver.restore(sess,tf.train.latest_checkpoint('./'))

# WとBを復元
graph = tf.get_default_graph()
weight = graph.get_tensor_by_name("wc1:0")
bias = graph.get_tensor_by_name("bc1:0")
sess.run(weight)
# 画像を加工して復元したモデルに渡せるか確認
"""
