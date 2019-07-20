from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np

input_data = 'testimage.jpg'

IMAGE_SIZE = 64

# モデルの読み込み
model = model_from_json(open('DL2parameter.json', 'r').read())

# 重みの読み込み
model.load_weights('DL2parameter.h5')

img = load_img(input_data, target_size=(IMAGE_SIZE,IMAGE_SIZE))
img_array  = img_to_array(img)
img_array = np.asarray(img_array)
img_array = img_array.astype('float32')
img_array = img_array / 255.0

# なぜか画像の予測ができない
# 読み込んだ学習済みモデルで予測
y = model.predict(np.array([img_array]))
print(y) # [[ 0.17429274]]
