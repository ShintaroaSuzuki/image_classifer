from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

# 学習用データ
train = './data/_train/data.txt'
# 検証用データ
test = './data/_test/data.txt'

# 学習訓練の試行回数
epochs = 30
# 1回の学習で何枚の画像を使うか
batch_size = 50
# 学習率、小さすぎると学習が進まないし、大きすぎても誤差が収束しなかったり発散したりしてダメとか。繊細
learning_rate = 1e-04

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

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

# create some convolutional layers
layer1 = create_new_conv_layer(x, 3, 16, [3, 3], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 16, 32, [3, 3], [2, 2], name='layer2')

flattened = tf.reshape(layer2, [-1, 16 * 16 * 32])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([16 * 16 * 32, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, NUM_CLASSES], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([NUM_CLASSES], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
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
            _, c = sess.run([optimiser, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            avg_loss += c / total_batch
        test_acc = sess.run(accuracy,
                       feed_dict={x: test_image, y: test_label})
        print("Epoch:", (epoch + 1), "loss =", "{:.4f}".format(avg_loss), "test accuracy: {:.4f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: test_image, y: test_label}))
