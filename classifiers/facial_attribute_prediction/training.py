import os
import cv2
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from keras.models import Model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from ..facial_attribute_prediction.data_preprocessing import CelebA

# loads all attributes (40)
celeba = CelebA()

"""# 2. Model Architecture
---
* MobileNetV2: as the base architecture pre-trained on 'imagenet'
* Summary: layers, num of parameters.
"""

# @title Model Input
img_size = 224  # @param ["192", "224"] {type:"raw", allow-input: true}

IMG_W = img_size
IMG_H = img_size
IMG_SHAPE = (IMG_H, IMG_W, 3)
TARGET_SIZE = (IMG_H, IMG_W)


# the architecture:

def mobilenet_model(num_features):
    base = MobileNetV2(input_shape=IMG_SHAPE,
                       weights=None,
                       include_top=False,
                       pooling='avg')

    # model top
    x = base.output
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    top = Dense(num_features, activation='sigmoid')(x)

    return Model(inputs=base.input, outputs=top)


model = mobilenet_model(num_features=celeba.num_features)
model.summary()

"""# 3. Training
---
* Data generators (for training and validation)
* Image Augmentation: zoom, rotation, shear, shift.
* Optimizer
* Checkpointing
* Fitting
"""

# Training Parameters
batch_size = 80  # @param ["64", "80", "96", "128"] {type:"raw", allow-input: true}
num_epochs = 12  # @param ["8", "16", "32"] {type:"raw", allow-input: true}

# ------------------------------------------------------------------------------
# -- Preparing Data Generators for training and validation set
# ------------------------------------------------------------------------------

# data augmentation only for the training istances:
train_datagen = ImageDataGenerator(rotation_range=20,
                                   rescale=1. / 255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1. / 255)

# get training and validation set:
train_split = celeba.split('training', drop_zero=False)[:500]
valid_split = celeba.split('validation', drop_zero=False)[:100]

# data generators:
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=TARGET_SIZE,
    batch_size=batch_size,
    class_mode='other'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_split,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=TARGET_SIZE,
    batch_size=batch_size,
    class_mode='other'
)


# ------------------------------------------------------------------------------
# -- Example of the augmented samples:
# ------------------------------------------------------------------------------

def load_reshape_img(path, shape=IMG_SHAPE):
    img = load_img(path, target_size=shape)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)
    return x


# select and load a picture:
sample_path = os.path.join(celeba.images_folder, train_split.sample(1).index[0])
sample = load_reshape_img(sample_path)

# plot ten augmented images
plt.figure(figsize=(20, 10))
plt.suptitle('Data Augmentation Example', fontsize=28)

for i, image in enumerate(train_datagen.flow(sample, batch_size=1)):
    if i == 10:
        break

    plt.subplot(2, 5, i + 1)
    plt.grid(False)
    plt.imshow(image.reshape(IMG_SHAPE) * 255.)

_ = plt.show()

# ------------------------------------------------------------------------------
# -- Compile model
# ------------------------------------------------------------------------------

model.compile(loss='cosine_proximity',
              optimizer='adadelta',  # adadelta, adam, nadam
              metrics=['binary_accuracy'])

# # ------------------------------------------------------------------------------
# # -- Checkpointing: at each epoch, the best model so far is saved
# # ------------------------------------------------------------------------------
#
# # model_path = f"{save_path}/UL19/weights-FC37-MobileNetV2-" + "{val_binary_accuracy:.2f}.hdf5"
# model_path = f"{save_path}/UL19/weights-FC40-MobileNetV2" + "{val_binary_accuracy:.2f}.hdf5"
#
# checkpoint = ModelCheckpoint(
#     model_path,
#     monitor='val_binary_accuracy',
#     save_best_only=True,
#     mode='max',
#     verbose=1
# )

# ------------------------------------------------------------------------------
# -- Fitting
# ----------------------------------------------------------------------------
history = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    max_queue_size=1,
    shuffle=True,
    # callbacks=[checkpoint],
    verbose=1
)


def plot_model_history(history):
    '''plots useful graphs about the model training: loss, accuracy, ecc.'''
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# plot the training and validation loss for every epoch:
plot_model_history(history)

"""# 4. Testing
---
* Test Data Generator
* Evaluate performance on test-set
* Average Hamming Distance, Mis-predictions
* Plot Mis-predictions frequency, and accuracy per feature
"""

# optionally load a pre-trained model:

model = keras.models.load_model(f"{save_path}/UL19/weights-FC37-MobileNetV2-0.92.hdf5")

# ------------------------------------------------------------------------------
# -- Data Generator for Test-Set
# ------------------------------------------------------------------------------

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = celeba.split('test', drop_zero=False)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_set,
    directory=celeba.images_folder,
    x_col='image_id',
    y_col=celeba.features_name,
    target_size=TARGET_SIZE,
    batch_size=batch_size,
    class_mode='other'
)

# ------------------------------------------------------------------------------
# -- Evaluate Accuracy
# ------------------------------------------------------------------------------

score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator),
    max_queue_size=1,
    verbose=1
)

print("Test score:", score[0])
print("Test accuracy:", score[1])


def hamming_distance(x, y):
    '''Hamming distance: use to get the number of mis-predicted features'''
    assert (len(x) == len(y))

    count = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            count += 1

    return count


def count_mistakes(preds, labels):
    '''For every feature counts the number mis-predictions'''
    mistakes = np.zeros(len(preds[0]))

    for i, pred in enumerate(preds):
        label = labels[i]

        for j in range(len(pred)):
            if pred[j] != label[j]:
                mistakes[j] += 1

    return mistakes


# ------------------------------------------------------------------------------
# -- Avg. Hamming Distance, Mis-predictedion frequency
# ------------------------------------------------------------------------------

mistakes = np.zeros(celeba.num_features)
count = 0
avg_d = 0
show_prob = 0.01
times_one = np.zeros(celeba.num_features)

# generate samples (x), and labels (y) a batch at a time:
for i in range(len(test_generator)):
    x, y = next(test_generator)
    preds = model.predict_on_batch(x)
    preds = np.round(preds).astype('int')  # make the prediction a binary vector
    count += len(x)

    mistakes += count_mistakes(preds, y)

    for i, p in enumerate(preds):
        d = hamming_distance(p, y[i])
        avg_d += d
        times_one += p

        if random.random() <= show_prob:
            # prints: prediction, true labels, and their hamming distance
            print(f"pred: {p}")
            print(f"true: {y[i]}")
            print(f"=> d = {d}")

# get average hamming distance:
avg_d /= count
print("avg. hamming: " + str(avg_d))

mistakes2 = []
for i, times in enumerate(times_one):
    val = mistakes[i] / times
    mistakes2.append(round(val, 2))

mistakes = np.round(mistakes / count, 2)

# plot mis-predicted feature ratio

plt.figure(figsize=(12, 12))

plt.barh(celeba.features_name, mistakes, color='red')
plt.title("Mis-Prediction Ratio")
plt.grid(False)

_ = plt.show()

# plot mis-predicted feature ratio

plt.figure(figsize=(12, 12))

plt.barh(celeba.features_name, mistakes2 / max(mistakes2), color='blue')
plt.title("Mis-Prediction Ratio")
plt.grid(False)

_ = plt.show()

# plot feature accuracy
plt.figure(figsize=(12, 12))

plt.barh(celeba.features_name, 1 - mistakes, color='green')
plt.title("Feature Accuracy")
plt.grid(False)

_ = plt.show()

# plot mis-predicted feature ratio

plt.figure(figsize=(12, 12))

plt.barh(celeba.features_name, 1 - (mistakes2 / max(mistakes2)), color='orange')
plt.title("Mis-Prediction Ratio")
plt.grid(False)

_ = plt.show()
