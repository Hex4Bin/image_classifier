# Import libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# Load a pre-defined dataset (70k of 28x28)
fashion_mnist = keras.datasets.fashion_mnist

# pull out data from dataset
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()


# Define our neural network structure
model = keras.Sequential([
    # input is a 28x28 image("Flatten" flattens the 28x28 into a single 784x1 input layer)
    keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden layer is 128 deep. relu returns the value or 0 if the number < 0 (works good enough, much faster)
    keras.layers.Dense(units=64, activation=tf.nn.relu),

    keras.layers.Dense(units=64, activation=tf.nn.relu),

    # output is 0-10 (depending on what piece of clothing it is) return maximum
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# Compile our moduel
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train our model, using our training data
model.fit(train_images, train_labels, epochs=5)

# Test our model, using our training data
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(test_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

print('Test data: ' + str(test_labels[0]))

# Make predictions
predictions = model.predict(test_images)

# Print out predictions
print('Predicted data: ' +
      str(list(predictions[0]).index(max(predictions[0]))))
