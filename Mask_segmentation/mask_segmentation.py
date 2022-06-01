#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout, UpSampling2D, Input
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TRAIN_PATH = 'aster_dataset/segmentation/train/'
VAL_PATH   = 'aster_dataset/segmentation/val/'
#TEST_PATH  = 'aster_dataset/segmentation/test/'

WEIGHTS_DIR = "callbacks/segmentation/weights/"
TB_DIR      = "callbacks/segmentation/tensorboard/"
CSV_DIR     = "callbacks/segmentation/csvlogger/"
BEST_DIR    = "callbacks/segmentation/best_model/"

SIZE = 256
SEED = 42
BATCH_SIZE = 6
EPOCHS = 50
STEPS_VAL = 15 * BATCH_SIZE
STEPS_TRAIN = 150 * BATCH_SIZE
LR = 0.0002

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[ ]:





# In[2]:


class AugmentedDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, color_mode='grayscale', *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            ori = next(generator)
            
            gc.collect()
            yield ori

            
base_train_generator = AugmentedDataGenerator(
    horizontal_flip=True
)

train_generator_image = base_train_generator.flow_from_directory(
    TRAIN_PATH + 'images_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13d'],
    seed = SEED
)

train_generator_mask = base_train_generator.flow_from_directory(
    TRAIN_PATH + 'masks_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13d'],
    seed = SEED
)

#train_generator = zip(train_image_generator, train_mask_generator)


base_val_generator = AugmentedDataGenerator()
val_generator_image = base_val_generator.flow_from_directory(
    VAL_PATH + 'images_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13d'],
    seed = SEED
)

val_generator_mask = base_val_generator.flow_from_directory(
    VAL_PATH + 'masks_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13d'],
    seed = SEED
)

#val_generator = zip(val_image_generator, val_mask_generator)

"""
base_test_generator = AugmentedDataGenerator()
test_generator_image = base_test_generator.flow_from_directory(
    TEST_PATH,
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['images_extended'],
    seed = SEED
)

test_generator_mask = base_test_generator.flow_from_directory(
    TEST_PATH,
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['masks_extended'],
    seed = SEED
)

#test_generator = zip(test_image_generator, test_mask_generator)

image_generator = base_test_generator.flow_from_directory(
    'aster_dataset/segmentation/val/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['images_extended']
)
"""

def image_and_mask_generator(image_generator, mask_generator):
    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        
        img[img < 0] = 0
        img_max = np.amax(img)
        img_min = np.amin(img)
        img = (img - img_min) / (img_max - img_min)

        gc.collect()
        yield img, mask


train_generator = image_and_mask_generator(train_generator_image, train_generator_mask)        
val_generator   = image_and_mask_generator(val_generator_image, val_generator_mask)
#test_generator  = image_and_mask_generator(test_generator_image, test_generator_mask)


# In[3]:


images, masks = next(train_generator)
print("image shape: ", images.shape)
print("mask  shape: ", masks.shape)


# In[4]:


def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


# In[5]:


#MODEL_WEIGHTS = WEIGHTS_DIR + 'weights_segmentation_multi_sigmoid_dropout02_batchnorm.40-0.00-0.00.hdf5'

model = UNet((SIZE, SIZE, 1), start_ch=64, dropout=0.5, batchnorm=True)
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics='accuracy')
#model.load_weights(MODEL_WEIGHTS)
model.summary()


# In[3]:


class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()


# In[7]:


history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_TRAIN,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=STEPS_VAL,
    callbacks=[
        CSVLogger(CSV_DIR + "log.csv", append=True),
        TensorBoard(log_dir=TB_DIR, write_graph=True),
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint(BEST_DIR + "segmentation_rgi13d" + ".hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True),
        ModelCheckpoint(WEIGHTS_DIR + "weights_segmentation_rgi13d.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
        MemoryCallback()
    ]
)


# In[8]:


plt.figure(figsize=(8,8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="Best model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('data/learning_curve/segmentated_rgi13d.png')
plt.show()

print("Best epoch: ", np.argmin(history.history["val_loss"]))


# In[9]:


images, masks = next(val_generator)
pred = model.predict(images)


# In[10]:


# Show side by side
_, axes = plt.subplots(BATCH_SIZE, 4, figsize=(20, 20))
for i in range(BATCH_SIZE):
    axes[i,0].imshow(images[i],  cmap='terrain', vmin=0)
    axes[i,1].imshow(masks[i], cmap='gray_r')
    axes[i,2].imshow(pred[i],   cmap='gray_r')
    axes[i,3].imshow(np.where(pred[i] > np.max(pred[i])/2, 1, 0),   cmap='gray_r')
    axes[i,0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,2].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,3].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
      
axes[0,0].set_title('Original image')
axes[0,1].set_title('Mask')
axes[0,2].set_title('Prediction')
axes[0,3].set_title('Threshold')
    
plt.tight_layout()
plt.savefig("figures/segmentation_rgi13d")
plt.show()


# In[5]:


base_train_generator = AugmentedDataGenerator(
    horizontal_flip=True
)

train_generator_image = base_train_generator.flow_from_directory(
    TRAIN_PATH + 'images_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13e'],
    seed = SEED
)

train_generator_mask = base_train_generator.flow_from_directory(
    TRAIN_PATH + 'masks_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13e'],
    seed = SEED
)

#train_generator = zip(train_image_generator, train_mask_generator)


base_val_generator = AugmentedDataGenerator()
val_generator_image = base_val_generator.flow_from_directory(
    VAL_PATH + 'images_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13e'],
    seed = SEED
)

val_generator_mask = base_val_generator.flow_from_directory(
    VAL_PATH + 'masks_extended/',
    target_size = (SIZE, SIZE),
    batch_size = BATCH_SIZE,
    classes = ['rgi13e'],
    seed = SEED
)

train_generator = image_and_mask_generator(train_generator_image, train_generator_mask)        
val_generator   = image_and_mask_generator(val_generator_image, val_generator_mask)


# In[6]:


model = UNet((SIZE, SIZE, 1), start_ch=64, dropout=0.5, batchnorm=True)
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics='accuracy')


# In[8]:


history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_TRAIN,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=STEPS_VAL,
    callbacks=[
        CSVLogger(CSV_DIR + "log.csv", append=True),
        TensorBoard(log_dir=TB_DIR, write_graph=True),
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint(BEST_DIR + "segmentation_rgi13e" + ".hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True),
        ModelCheckpoint(WEIGHTS_DIR + "weights_segmentation_rgi13e.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
        MemoryCallback()
    ]
)


# In[9]:


plt.figure(figsize=(8,8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="Best model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('data/learning_curve/segmentated_rgi13e.png')
plt.show()

print("Best epoch: ", np.argmin(history.history["val_loss"]))


# In[10]:


images, masks = next(val_generator)
pred = model.predict(images)


# In[11]:


# Show side by side
_, axes = plt.subplots(BATCH_SIZE, 4, figsize=(20, 20))
for i in range(BATCH_SIZE):
    axes[i,0].imshow(images[i],  cmap='terrain', vmin=0)
    axes[i,1].imshow(masks[i], cmap='gray_r')
    axes[i,2].imshow(pred[i],   cmap='gray_r')
    axes[i,3].imshow(np.where(pred[i] > np.max(pred[i])/2, 1, 0),   cmap='gray_r')
    axes[i,0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,2].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,3].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
      
axes[0,0].set_title('Original image')
axes[0,1].set_title('Mask')
axes[0,2].set_title('Prediction')
axes[0,3].set_title('Threshold')
    
plt.tight_layout()
plt.savefig("figures/segmentation_rgi13e")
plt.show()


# In[ ]:




