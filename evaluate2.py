import tensorflow as tf

# TensorFlowのセッション
sess = tf.Session()

# 訓練済みモデルのmetaファイルを読み込み
saver = tf.train.import_meta_graph('model.ckpt.meta')

# モデルの復元
saver.restore(sess,tf.train.latest_checkpoint('./'))

# WとBを復元
graph = tf.get_default_graph()
weight = graph.get_tensor_by_name("wc1:0")
bias = graph.get_tensor_by_name("bc1:0")

# 画像を加工して復元したモデルに渡せるか確認
