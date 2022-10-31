import pandas as pd
import numpy as np
import cv2
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# if u try on jupyter notebook, tf.keras can be used too...the reason using keras is bcoz of error in vs code

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def visualize_data_count(data):            #to visualize data with countplot
    p=[]
    for face in data:
        if (face[1] == 0 ):
            p.append('Mask')
        else:
            p.append('No Mask')
    sns.countplot(p)

def getJson(file_path):                     #to load json file format
    with open(file_path, 'r') as f:
        return json.load(f)

def adjust_brightness(image, gamma =1.0):     #gamma 1.0 for default
    invGamma = 1.0/gamma
    table = np.array([((i/255.0) ** invGamma)* 255 for i in np.arange(0,256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def loading_dataset(img_size =124):
    '''jsonfiles = []
    for i in os.listdir(annotation_directory):
        jsonfiles.append(getJson(os.path.join(annotation_directory, i))) '''
    data = []
    mask = ['face_with_mask']
    non_mask = ['face_no_mask']
    labels = {'mask': 0, 'without_mask': 1}
    for i in train_csv["name"].unique():
        f = i + ".json"

        for j in getJson(os.path.join(annotation_directory, f)).get("Annotations"):
            if j["classname"] in mask:
                x, y, w, h = j["BoundingBox"]
                img = cv2.imread(os.path.join(image_directory, i), 1)  # 1 for grayscale
                img = img[y: h, x: w]  # boundingbox for mask only
                img = cv2.resize(img, (img_size, img_size))
                data.append([img, labels["mask"]])

            if j["classname"] in non_mask:
                x, y, w, h = j["BoundingBox"]
                img = cv2.imread(os.path.join(image_directory, i), 1)
                img = img[y: h, x: w]
                img = cv2.resize(img, (img_size, img_size))
                data.append([img, labels["without_mask"]])
    random.shuffle(data)
    return data

def preprocessing_data(data):
    X = []
    Y = []
    for features, label in data:
        X.append(features)
        Y.append(label)
    X = np.array(X) / 255.0
    X = X.reshape(-1, 124, 124, 3)
    Y = np.array(Y)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, train_size=0.8, random_state=0)
    return x_train, x_val, y_train, y_val

def define_model():
    model = Sequential([
        ZeroPadding2D((1, 1), input_shape=(124, 124, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dropout(0.25),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(100, activation='relu'),
        Dropout(0.35),
        Dense(1, activation='sigmoid')

    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_evaluate(model):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(x_train)

    model.fit(datagen.flow(x_train, y_train, batch_size=48),
              steps_per_epoch=x_train.shape[0] // 48,
              epochs=10, verbose=1,
              validation_data=(x_val, y_val)
              )
    t_loss, t_acc = model.evaluate(x_val, y_val, verbose=2)
    print('\n Test_accuracy: ', t_acc)
    #model.save('mask_nomask.h5')   # to save trained model

def plot_tested_image():
    pat = r'D:/Data_collection/My_project_data/face_mask/deep_model/res10_300x300_ssd_iter_140000.caffemodel'   # add ur directory
    txtpat = r'D:/Data_collection/My_project_data/face_mask/deep_model/deploy.prototxt'
    cvvNet = cv2.dnn.readNetFromCaffe(txtpat, pat)

    test_images = ['1119.jpg', '1504.jpg', '0072.jpg', '0012.jpg', '0353.jpg', '1374.jpg']
    Gamma = 2.0
    fig = plt.figure(figsize=(14, 14))
    rows, cols = 3, 2
    axes = []
    assign = {'0': 'Mask', '1': 'No Mask'}

    for j, im in enumerate(test_images):
        image = cv2.imread(os.path.join(image_directory, im), 1)
        image = adjust_brightness(image, gamma=Gamma)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        cvvNet.setInput(blob)
        detections = cvvNet.forward()
        for i in range(0, detections.shape[2]):
            try:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                frame = image[startY: endY, startX: endX]
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    im = cv2.resize(frame, (img_size, img_size))
                    im = np.array(im) / 255.0
                    im = im.reshape(1, 124, 124, 3)
                    result = model.predict(im)
                    if result > 0.5:
                        label_Y = 1
                    else:
                        label_Y = 0
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(image, assign[str(label_Y)], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (40, 255, 20), 2)
            except:
                pass
        axes.append(fig.add_subplot(rows, cols, j + 1))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    img_size = 124

    annotation_directory = "D:/Data_collection/My_project_data/face_mask/face_mask/Medical mask/Medical mask/Medical Mask/annotations"
    # image_directory = "D:\linux_partation\data_machine\face_mask\Medical mask\Medical mask\Medical Mask\images"
    image_directory = "D:/Data_collection/My_project_data/face_mask/face_mask/Medical mask/Medical mask/Medical Mask/images"
    train_csv = pd.read_csv("D:/Data_collection/My_project_data/face_mask/face_mask/train.csv")
    test_csv = pd.read_csv("D:/Data_collection/My_project_data/face_mask/face_mask/submission.csv")

    data = loading_dataset()
    x_train, x_val, y_train, y_val = preprocessing_data(data)
    model = define_model()
    model_evaluate(model= model)
    plot_tested_image()