import tensorflow as tf
import numpy as np
import cv2
import os
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from math import sqrt
from sklearn.utils import shuffle

# 識別ラベルの数(今回は3つ)
NUM_CLASSES = 3
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 64
# LeakyReLUのパラメータ
alpha = 0.15

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

# 学習用データ
train = './data/_train/data.txt'
# 検証用データ
test = './data/_test/data.txt'
# tensorboradの保存先
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

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
    test_image, test_label = shuffle(test_image, test_label)
    test_image.append(img_to_array(load_img("testimage0.jpg", target_size=(IMAGE_SIZE,IMAGE_SIZE))))
    test_image = np.asarray(test_image)
    test_image = test_image.astype('float32')
    test_image = test_image / 255.0
    test_label = np.asarray(test_label)
    test_label = np_utils.to_categorical(test_label, NUM_CLASSES)

tf.reset_default_graph()

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from
# mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
# now declare the output data placeholder - 3 digits
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# batch_normalization_bool
is_training = tf.placeholder(dtype=bool, name='is_training')

# BatchNormalization
def batch_norm(X, axes, shape, is_training):
    """
    バッチ正規化
    平均と分散による各レイヤの入力を正規化(白色化)する
    """
    if is_training is False:
        return X
    epsilon = 1e-5
    # 平均と分散
    mean, variance = tf.nn.moments(X, axes)
    # scaleとoffsetも学習対象
    scale = tf.Variable(tf.ones([shape]))
    offset = tf.Variable(tf.zeros([shape]))
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)

# create some convolutional layers
wc1 = tf.Variable(tf.truncated_normal([3 ,3 ,3, 16], stddev=0.03), name='wc1')
bc1 = tf.Variable(tf.truncated_normal([16], name='bc1'))
conv_layer1 = tf.nn.conv2d(x, wc1, [1, 1, 1, 1], padding='SAME')
conv_layer1 += bc1
conv_layer1 = batch_norm(conv_layer1, [0,1,2], 16, is_training)
conv_layer1 = tf.nn.relu(conv_layer1)
conv_layer1 = tf.nn.max_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

wc2 = tf.Variable(tf.truncated_normal([3 ,3 ,16, 32], stddev=0.03), name='wc2')
bc2 = tf.Variable(tf.truncated_normal([32], name='bc2'))
conv_layer2 = tf.nn.conv2d(conv_layer1, wc2, [1, 1, 1, 1], padding='SAME')
conv_layer2 += bc2
conv_layer2 = batch_norm(conv_layer2, [0,1,2], 32, is_training)
conv_layer2 = tf.nn.relu(conv_layer2)
conv_layer2 = tf.nn.max_pool(conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

wc3 = tf.Variable(tf.truncated_normal([3 ,3 ,32, 64], stddev=0.03), name='wc3')
bc3 = tf.Variable(tf.truncated_normal([64], name='bc3'))
conv_layer3 = tf.nn.conv2d(conv_layer2, wc3, [1, 1, 1, 1], padding='SAME')
conv_layer3 += bc3
conv_layer3 = batch_norm(conv_layer3, [0,1,2], 64, is_training)
conv_layer3 = tf.nn.relu(conv_layer3)
conv_layer3 = tf.nn.max_pool(conv_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

wc4 = tf.Variable(tf.truncated_normal([3 ,3 ,64, 128], stddev=0.03), name='wc4')
bc4 = tf.Variable(tf.truncated_normal([128], name='bc4'))
conv_layer4 = tf.nn.conv2d(conv_layer3, wc4, [1, 1, 1, 1], padding='SAME')
conv_layer4 += bc4
conv_layer4 = batch_norm(conv_layer4, [0,1,2], 128, is_training)
conv_layer4 = tf.nn.relu(conv_layer4)
conv_layer4 = tf.nn.max_pool(conv_layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flattened = tf.reshape(conv_layer4, [-1, 4 * 4 * 128])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([4 * 4 * 128, 1024], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = batch_norm(dense_layer1, [0], 1024, is_training)
dense_layer1 = tf.nn.leaky_relu(dense_layer1, alpha=alpha)
drop_prob1 = tf.placeholder(tf.float32)
dense_layer1 = tf.nn.dropout(dense_layer1, rate=drop_prob1)

# add dence layer
wd2 = tf.Variable(tf.truncated_normal([1024, 256], stddev=sqrt(2 / 1024)), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([256], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
dense_layer2 = batch_norm(dense_layer2, [0], 256, is_training)
dense_layer2 = tf.nn.leaky_relu(dense_layer2, alpha=alpha)
drop_prob2 = tf.placeholder(tf.float32)
dense_layer2 = tf.nn.dropout(dense_layer2, rate=drop_prob2)

wd3 = tf.Variable(tf.truncated_normal([256, NUM_CLASSES], stddev=sqrt(2 / 256)), name='wd3')
bd3 = tf.Variable(tf.truncated_normal([NUM_CLASSES], stddev=0.01), name='bd3')
dense_layer3 = tf.matmul(dense_layer2, wd3) + bd3
y_ = tf.nn.softmax(dense_layer3)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, 'model.ckpt')

print(sess.run(y_, feed_dict={x: test_image, y: test_label, drop_prob1: 0, drop_prob2: 0, is_training: False})[-1])

# 渡すデータの数によって予測結果が変わってしまう。テストデータの順序に予測が依存してしまっているわけではない。

# テストデータの最後に付け足すような形にするとうまく予測できる
