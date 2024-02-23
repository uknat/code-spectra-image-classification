# To develop an image classification project using TensorFlow that meets the outlined expectations and 
# solves the stated problem, we will go through each phase step-by-step, ensuring we cover data collection
# and preparation, exploratory data analysis (EDA), model selection and architecture, model training, 
# evaluation, testing, and deployment. This guide provides a high-level overview and example code snippets
# to get you started.

# STEP 1. Data Collection and Preparation
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten labels for easier processing
train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

# STEP 2. Exploratory Data Analysis (EDA)
# Display some images from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Distribution of classes
plt.figure(figsize=(8, 4))
sns.countplot(x=train_labels)
plt.title('Distribution of Classes')
plt.show()

# STEP 3. Model Selection and Architecture
# Here, we create a simple Convolutional Neural Network (CNN) model suitable for classifying images from 
# the CIFAR-10 dataset.


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.summary()

# STEP 4. Model Training
# Compile and train the model.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# STEP 5. Model Evaluation
# Evaluate the model's performance on the test dataset.

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# STEP 6. Model Testing
# Make predictions with the trained model.


predictions = model.predict(test_images)

# STEP 7. Deployment
# For deploying the model, you would first save it and then use TensorFlow Serving, TensorFlow Lite, or 
# another deployment solution suitable for your requirements.


# Save the model
model.save('my_cifar10_model.h5')

