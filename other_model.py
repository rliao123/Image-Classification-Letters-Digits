import os
import matplotlib.pyplot as plt
import ssl
import tensorflow as tf
from keras.applications import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
ssl._create_default_https_context = ssl._create_unverified_context

dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dir, 'dataset')
train_path = os.path.join(path, 'train')

# load data: training set (80%), validation set (10%), test set (10%)
train_data = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="training",
    seed=100, image_size=(128, 128), batch_size=32, label_mode='categorical')
val_path = os.path.join(path, 'validation')
val_data = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.1, subset="validation",
    seed=100, image_size=(128, 128), batch_size=32, label_mode='categorical')
test_path = os.path.join(path, 'test')
test_data = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.1, subset="validation",
    seed=100, image_size=(128, 128), batch_size=32, label_mode='categorical')

num_of_classes = 36

model = Sequential()
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
inception.trainable = False
model.add(inception)
model.add(Flatten())
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, epochs=80, validation_data=val_data)

# visualize accuracy, validation accuracy, loss
accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
loss = history.history['loss']
validation_loss = history.history['val_loss']

e = range(len(accuracy))

plt.title('Training and Validation Accuracy')
plt.plot(e, accuracy, 'r', label='Training Accuracy')
plt.plot(e, validation_accuracy, 'g', label='Validation Accuracy')
plt.legend()
plt.figure()

plt.title('Training and Validation Loss')
plt.plot(e, loss, 'r', label='Training Loss')
plt.plot(e, validation_loss, 'g', label='Validation Loss')
plt.legend()

# use model with test set
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

plt.show()
