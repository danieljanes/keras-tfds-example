from typing import Tuple
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow_datasets as tfds


LOG_DIR: str = "logs"
EPOCHS: int = 5
BATCH_SIZE: int = 32


def main() -> None:
  # Dataset: Use either ds_real() or ds_random()
  # - ds_mnist: provides a tf.data.Dataset after downloading MNIST
  # - ds_rndm:  provides a tf.data.Dataset of the same shape, w/o any download
  ds_train, ds_test, num_classes, m_train, m_test = ds_rndm()

  STEPS_PER_EPOCH: int = int(m_train / BATCH_SIZE)

  # Zero-pad images to make them compatible with the LeNet-5 architecture
  ds_train = ds_train.map(preprocessing)
  ds_test = ds_test.map(preprocessing)

  # Training
  optimizer = tf.train.AdamOptimizer()
  train(ds_train, ds_test, m_train, m_test, num_classes, BATCH_SIZE, STEPS_PER_EPOCH, optimizer)


def train(ds_train, ds_test, m_train, m_test, num_classes, batch_size, steps_per_epoch, optimizer):
  ds_train = ds_train.repeat().shuffle(buffer_size=10000).batch(batch_size)
  ds_test = ds_test.batch(m_test)

  # Build model
  model = build_model(num_classes)
  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  print(model.summary())

  # Training
  tb_callback = callbacks.TensorBoard(LOG_DIR)
  history = model.fit(ds_train, epochs=EPOCHS,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[tb_callback])
  print(history)

  # Evaluation
  score = model.evaluate(ds_test, steps=1)
  print("Test set loss:    ", score[0])
  print("Test set accuracy:", score[1])


def ds_mnist() -> Tuple[Dataset, Dataset, int, int, int]:
  # Download and extract dataset using TFDS
  (ds_train, ds_test), info = tfds.load(name="mnist",
                                        split=["train", "test"],
                                        as_supervised=True,
                                        with_info=True)
  # Number of classes, number of training/test examples
  num_classes: int = info.features['label'].num_classes
  m_train: int = info.splits['train'].num_examples
  m_test: int = info.splits['test'].num_examples

  return ds_train, ds_test, num_classes, m_train, m_test


def ds_rndm() -> Tuple[Dataset, Dataset, int, int, int]:
  # Hardcoded values taken from MNIST
  num_classes = 10
  m_train = 60000
  m_test = 10000
  # Random noise
  ds_image = Dataset.from_tensor_slices((
              tf.random_uniform([m_train, 28, 28, 1], maxval=255, dtype=tf.int32)
             ))
  ds_label = Dataset.from_tensor_slices((
              tf.random_uniform([m_train], maxval=9, dtype=tf.int64)
             ))
  ds_train = Dataset.zip((ds_image, ds_label))
  ds_test = ds_train.take(m_test)

  return ds_train, ds_test, num_classes, m_train, m_test


def preprocessing(x, y):
  x = tf.image.pad_to_bounding_box(x, 2, 2, 32, 32)
  x = tf.cast(x, tf.float32)
  x = x / 255
  y = tf.one_hot(y, 10)
  return x, y


def build_model(num_classes: int) -> tf.keras.Model:
  inputs = tf.keras.Input(shape=(32, 32, 1))
  
  x = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1))(inputs)
  x = layers.Activation('tanh')(x)
  x = layers.AveragePooling2D(strides=(2, 2))(x)

  x = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1))(x)
  x = layers.Activation('tanh')(x)
  x = layers.AveragePooling2D(strides=(2, 2))(x)

  x = layers.Flatten()(x)
  x = layers.Dense(120, activation='tanh')(x)
  x = layers.Dense(84, activation='tanh')(x)
  outputs = layers.Dense(num_classes, activation='softmax')(x)
  
  return tf.keras.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
  main()
