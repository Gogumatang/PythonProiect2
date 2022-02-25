import os
import numpy as np
from tqdm import tqdm

from PIL import Image
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


testdir = "./data/seg_test"
trainsetY = []

traindir = "data/seg_test/buildings"
dir = os.listdir(traindir)

traindir = "data/seg_test/buildings/" + dir[0]
imagee = Image.open(traindir,mode='r')
img_tensor = image.img_to_array(imagee)
img_tensor /= 255.
trainsetX = img_tensor
trainsetX = np.reshape(trainsetX,(1,150,150,3))
trainsetY = []



for i in tqdm(range(len(dir))):
    traindir = "data/seg_test/buildings/" + dir[i]
    imagee = Image.open(traindir,mode='r')
    img_tensor = image.img_to_array(imagee)
    img_tensor = img_tensor / 255
    img_tensor = np.reshape(img_tensor,(1,150,150,3))
    trainsetX = np.append(trainsetX, img_tensor, axis=0)

print(trainsetX.shape)


for i in range(len(dir)):
    trainsetY.append(0)

traindir = "data/seg_test/glacier"
dir = os.listdir(traindir)
for i in tqdm(range(len(dir))):
    traindir = "data/seg_test/glacier/" + dir[i]
    imagee = Image.open(traindir, mode='r')
    img_tensor = image.img_to_array(imagee)
    img_tensor = img_tensor / 255
    if(np.size(img_tensor)!=67500):
        print(dir[i])
    else:
        img_tensor = np.reshape(img_tensor, (1, 150, 150, 3))
        trainsetX = np.append(trainsetX, img_tensor, axis=0)

for i in range(len(dir)):
    trainsetY.append(1)

trainsetX = np.array(trainsetX)
trainsetY = np.array(trainsetY)

model = Sequential()
model.add(Conv2D(32, (7, 7), padding="valid", input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainsetX, trainsetY, epochs=6, batch_size=10)

model.save('771_model.h5')

