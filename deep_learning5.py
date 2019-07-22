# 層の数やハイパーパラメータを調整したり、DropoutやBatchNormalizationを行ったり、He初期化を行ったり、活性化関数を工夫したりして精度を上げる。

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from math import sqrt
import os

# 学習用データ
train = './data/_train/data.txt'
# 検証用データ
test = './data/_test/data.txt'
# tensorboradの保存先
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

# 学習訓練の試行回数
epochs = 300
# 1回の学習で何枚の画像を使うか
batch_size = 50
# 学習率、小さすぎると学習が進まないし、大きすぎても誤差が収束しなかったり発散したりしてダメとか。繊細
learning_rate = 1e-04
# LeakyReLUのパラメータ
alpha = 0.15

# parameter of dropout
drop_rate1 = 0.7
drop_rate2 = 0.7

# 識別ラベルの数(今回は3つ)
NUM_CLASSES = 3
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 64

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
wc1 = tf.Variable(tf.truncated_normal([3 ,3 ,3, 16], stddev=sqrt(2 / 3)), name='wc1')
bc1 = tf.Variable(tf.truncated_normal([16], name='bc1'))
conv_layer1 = tf.nn.conv2d(x, wc1, [1, 1, 1, 1], padding='SAME')
conv_layer1 += bc1
conv_layer1 = batch_norm(conv_layer1, [0,1,2], 16, is_training)
conv_layer1 = tf.nn.relu(conv_layer1)
conv_layer1 = tf.nn.max_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

wc2 = tf.Variable(tf.truncated_normal([3 ,3 ,16, 32], stddev=sqrt(2 / 16)), name='wc2')
bc2 = tf.Variable(tf.truncated_normal([32], name='bc2'))
conv_layer2 = tf.nn.conv2d(conv_layer1, wc2, [1, 1, 1, 1], padding='SAME')
conv_layer2 += bc2
conv_layer2 = batch_norm(conv_layer2, [0,1,2], 32, is_training)
conv_layer2 = tf.nn.relu(conv_layer2)
conv_layer2 = tf.nn.max_pool(conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

wc3 = tf.Variable(tf.truncated_normal([3 ,3 ,32, 16], stddev=sqrt(2 / 32)), name='wc3')
bc3 = tf.Variable(tf.truncated_normal([16], name='bc3'))
conv_layer3 = tf.nn.conv2d(conv_layer2, wc3, [1, 1, 1, 1], padding='SAME')
conv_layer3 += bc3
conv_layer3 = batch_norm(conv_layer3, [0,1,2], 16, is_training)
conv_layer3 = tf.nn.relu(conv_layer3)
conv_layer3 = tf.nn.max_pool(conv_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flattened = tf.reshape(conv_layer3, [-1, 8 * 8 * 16])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([8 * 8 * 16, 1024], stddev=sqrt(2 / 16)), name='wd1')
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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer3, labels=y))

# add an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

history = {
    'acc': [],
    'val_acc': []
}

with tf.Session() as sess:
    tf.summary.FileWriter(LOG_DIR, sess.graph)  # tensorboard
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(train_label) / batch_size)
    for epoch in range(epochs):
        X_, Y_ = shuffle(train_image, train_label)
        avg_loss= 0
        for i in range(total_batch):
            start = batch_size * i
            end = start + batch_size
            batch_x, batch_y = X_[start: end], Y_[start: end]
            _, c = sess.run([optimizer, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y, drop_prob1: drop_rate1, drop_prob2: drop_rate2, is_training: True})
            avg_loss += c / total_batch
        train_acc = sess.run(accuracy,
                       feed_dict={x: train_image, y: train_label, drop_prob1: drop_rate1, drop_prob2: drop_rate2, is_training: True})
        test_acc = sess.run(accuracy,
                       feed_dict={x: test_image, y: test_label, drop_prob1: 0, drop_prob2: 0, is_training: False})
        print("Epoch:", (epoch + 1), "loss =", "{:.4f}".format(avg_loss), "train accuracy: {:.4f}".format(train_acc), "test accuracy: {:.4f}".format(test_acc))
        history['acc'].append(train_acc)
        history['val_acc'].append(test_acc)

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: test_image, y: test_label, drop_prob1: 0, drop_prob2: 0}))

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()

# tensorboard --logdir=C:\Users\shinb\OneDrive\ドキュメント\_Python_Scripts\image_classifer\logs
