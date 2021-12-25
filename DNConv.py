#imports
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt

#creating convolutional base
model = models.Sequential()
model.add(layers.Conv2D(40, (3, 3), activation='relu', input_shape=(40, 80, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(80, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(80, (3, 3), activation='relu'))
#adding dense layers
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),#from_logits=True),
              metrics=['accuracy'])

#____________________________________________________________________________________________#

#Data generation from images
train_data_dir = 'data/train'
validation_data_dir = 'data/val'
batch_size = 32
epochs = 10
nb_train_samples = 21000
nb_validation_samples = 4500
#image dimensions
img_width, img_height = 80, 40

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#creating data generator for training - normalizes image pixel value
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

#using flow_from_directory to generate from pictures in the subfolders specified here 
#to generate batches of augmented image data - tho only training data is flipped etc.
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_height, img_width),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

#generator for validation set
validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,  # this is the target directory
        target_size=(img_height, img_width),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

#____________________________________________________________________________________________#

#fitting model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    use_multiprocessing=False, 
    workers=16)

#saving model weights
model.save_weights('DNConv_weights.h5')