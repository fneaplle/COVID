import numpy as np
import pandas as pd
import tensorflow as tf
import hparams

train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    hparams.DATA_PATH,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(hparams.IMG_HEIGHT, hparams.IMG_WIDTH),
    batch_size=hparams.BATCH_SIZE,
    #color_mode='grayscale'
)

val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    hparams.DATA_PATH,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(hparams.IMG_HEIGHT, hparams.IMG_WIDTH),
    batch_size=hparams.BATCH_SIZE,
    #color_mode='grayscale'
)

if __name__=='__main__':
    import sys, os
    from PIL import Image
    import matplotlib.pyplot as plt

    img=Image.open(hparams.DATA_PATH + 'covid' + '\\0C7E78DA-FAFC-480D-88B6-1459C51481AF-1068x817.jpeg')
    ds = np.array(img)
    print(ds.shape)


    #for img, label in train_ds.take(1):
    #    #plt.imshow(img[0])
    #    plt.imshow(img[0].numpy().astype('uint8'))
    #    break

