#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)


# In[3]:


VAL = 0.20


# In[4]:


import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

SEED = 1234
tf.random.set_seed(SEED)

cwd = os.getcwd()

import json
import shutil
import random

# Defining the datasets directory
dataset_dir = os.path.join(cwd, 'MaskDataset')
training_dir = os.path.join(dataset_dir, 'training')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

# Create validation directory if it doesn't exist
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

# Loading the classes of each image into the memory
train_classes_json_file_name = 'train_gt.json'
train_classes_json_directory = os.path.join(dataset_dir, train_classes_json_file_name)

data = {}

with open(train_classes_json_directory) as json_file:
    data = json.load(json_file)


# Creating folder for each class of image for training and validation datasets
classes = set(data.values())
print(classes)

for class_label in classes:
    class_training_dir = os.path.join(training_dir, str(class_label))
    class_validation_dir = os.path.join(validation_dir, str(class_label))
    if not os.path.exists(class_training_dir):
        os.makedirs(class_training_dir)
    if not os.path.exists(class_validation_dir):
        os.makedirs(class_validation_dir)

# Assigning images to each training folder/class, avoiding to have the same image two times in the same folder
for entry in os.scandir(training_dir):
    if(entry.is_file()):
        file_destination = os.path.join(training_dir, str(data[entry.name]), entry.name)
        if not os.path.isfile(file_destination):
            shutil.copy(entry.path, file_destination)
    
# Choosing random images to be into the validation folders, being able to repeat without cloning images
validation_rate = VAL

for class_label in classes:
    class_training_dir = os.path.join(training_dir, str(class_label))
    class_validation_dir = os.path.join(validation_dir, str(class_label))
    
    for old_entry in os.scandir(class_validation_dir):
        os.remove(old_entry.path)
    
    training_entries = list(os.scandir(class_training_dir))
    validation_size = round(len(training_entries)*validation_rate)
    
    for validation_entry in random.sample(training_entries, validation_size):
        destination = os.path.join(class_validation_dir, validation_entry.name)
        os.rename(validation_entry.path, destination)


# # Data augmentation

# In[5]:


apply_data_augmentation = True

if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=10,
        height_shift_range=10,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=0,
        rescale=1/255.
    )
else:
    train_data_gen = ImageDataGenerator(rescale=1/255.)

valid_data_gen = ImageDataGenerator(rescale=1/255.)
# test_data_gen = ImageDataGenerator(rescale=1/255.)

bs = 8

train_gen = train_data_gen.flow_from_directory(
    training_dir,
    batch_size=bs,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

valid_gen = valid_data_gen.flow_from_directory(
    validation_dir,
    batch_size=bs,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

# test_gen = test_data_gen.flow_from_directory(
#     test_dir,
#     batch_size=bs,
#     class_mode='categorical',
#     shuffle=True,
#     seed=SEED
# )

img_h = 256
img_w = 256

num_classes = len(classes)

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None, num_classes])
)

train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_generator(
    lambda: valid_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 256, 256, 3], [None, num_classes])
)

valid_dataset = valid_dataset.repeat()

# test_dataset = tf.data.Dataset.from_generator(
#     lambda: test_gen,
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([None, 256, 256, 3], [None, num_classes])
# )

# test_dataset = test_dataset.repeat()


# # Building the Network

# In[6]:


model = tf.keras.Sequential()


# In[7]:


start_f = 32
depth = 5

model = tf.keras.Sequential()
for i in range(depth):
    if i == 0:
        input_shape = [img_h, img_w, 3]
    else:
        input_shape = [None]

    # Convolutional part
    model.add(
        tf.keras.layers.Conv2D(
            filters=start_f,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="same",
            input_shape=input_shape
    ))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    start_f *= 2

# Fully connected part
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))


# In[8]:


model.summary()


# In[9]:


loss = tf.keras.losses.CategoricalCrossentropy()

lr = 1e-4

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# In[10]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', "--logdir 'D:\\Polimi\\Lectures\\3semester\\artificial-neural-networks-and-deep-learning\\Homeworks\\1st\\classification_experiments\\'")


# In[11]:


exps_dir = os.path.join(cwd, 'classification_experiments')
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'CNN'

exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(ckpt_dir, 'cp_.ckpt'), #'cp_{epoch:02d}.ckpt'
    save_weights_only=True # False to save the model directly
)
callbacks.append(ckpt_callback)

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=tb_dir,
    profile_batch=0,
    histogram_freq=1 # if 1 shows weights histograms
)
callbacks.append(tb_callback)

# Early Stopping
# --------------
early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(es_callback)


# In[12]:


model.fit(
    x=train_dataset,
    epochs=100,
    steps_per_epoch=len(train_gen),
    validation_data=valid_dataset,
    validation_steps=len(valid_gen),
    callbacks=callbacks
)


# # Read the CSV 

# In[15]:


import pandas as pd
def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


test_dir = os.path.join(dataset_dir, 'test')

images = [f for f in os.listdir(test_dir)]
images = pd.DataFrame(images)
images.rename(columns = {0:'filename'}, inplace = True)
images["class"] = 'test'

test_gen = train_data_gen.flow_from_dataframe(images,
                                               test_dir,
                                               batch_size=bs,
                                               target_size=(img_h, img_w),
                                               class_mode='categorical',
                                               shuffle=False,
                                               seed=SEED)


test_gen.reset()

predictions = model.predict_generator(test_gen, len(test_gen), verbose=1)

results = {}
images = test_gen.filenames
i = 0

for p in predictions:
  prediction = np.argmax(p)
  import ntpath
  image_name = ntpath.basename(images[i])
  results[image_name] = str(prediction)
  i = i + 1

create_csv(results,dataset_dir)


# In[14]:


dataset_dir

