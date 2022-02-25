import os
import numpy as np
from tqdm import tqdm

from PIL import Image
from keras.preprocessing import image
from keras.models import load_model

model = load_model('333_model.h5')
testdir = "data/seg_train/buildings"
dir = os.listdir(testdir)

testset = np.empty(shape=(1,150,150,3))

for i in tqdm(range(len(dir))):
    traindir = "data/seg_train/buildings/" + dir[i]
    imagee = Image.open(traindir, mode='r')
    img_tensor = image.img_to_array(imagee)
    img_tensor = img_tensor / 255
    if (np.size(img_tensor) != 67500):
        print(dir[i])
    else:
        img_tensor = np.reshape(img_tensor, (1, 150, 150, 3))
        testset = np.append(testset, img_tensor, axis=0)

testlabel = []

for i in range(len(dir)):
    testlabel.append(0)

testset = np.array(testset)
testlabel = np.array(testlabel)

loss, acc = model.evaluate(testset, testlabel)

print(str(loss) + ':오차율  ', str(acc) + ":정확도")