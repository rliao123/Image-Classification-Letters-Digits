import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import Sequential

dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dir, 'dataset')
train_path = os.path.join(path, 'train')

# load data: training set (80%), validation set (10%), test set (10%)
train_data = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="training",
    seed=100, image_size=(128, 128), batch_size=32)
val_data = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.1, subset="validation",
    seed=100, image_size=(128, 128), batch_size=32)
test_data = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.1, subset="validation",
    seed=100, image_size=(128, 128), batch_size=32)

class_names = train_data.class_names

'''
# visualize sample of images from training set
# plot a sample batch of images
plt.figure(figsize=(15, 15))
num_of_row = 9
num_of_col = 9
for images, labels in train_data.take(1):
    for i in range(min(len(class_names), len(images))):
        ax = plt.subplot(num_of_row, num_of_col, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


plt.show()
'''
# build model
model = Sequential()

# input layer
model.add(layers.Rescaling(1. / 255))  # normalize pixel values

# convolutional and pooling
model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # extract features
model.add(layers.MaxPool2D((2, 2)))  # reduce spatial dimensions
model.add(layers.Dropout(0.2))  # address overfitting

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((3, 3)))

# flatten and fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))

# output layer
model.add(layers.Dense(len(class_names), activation='softmax'))  # last layer: # of neurons = # of classes

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train model with training set and validate with validation set
history = model.fit(train_data,
                    validation_data=val_data, epochs=80)

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

# use model on test set
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

plt.show()
