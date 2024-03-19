import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from keras.models import Sequential

''' Loading Dataset '''
dataset = tfds.load('mnist')
train_orig, test_orig = dataset['train'], dataset['test']

''' Process Dataset '''
BUFFER_SIZE = 10000
BATCH_SIZE = 100
NUM_EPOCHS = 5

train = train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32))
)
test = test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32))
)
tf.random.set_seed(1)
train = train.shuffle(buffer_size=BUFFER_SIZE,
                      reshuffle_each_iteration=False)
val = train.take(10000).batch(BATCH_SIZE)
train = train.skip(10000).batch(BATCH_SIZE)


def Load_Model():
    return tf.keras.models.load_model('number_recog_model.keras')

def Train_Model(train, val):
    ''' LeNet Model '''
    model = Sequential([
        layers.Conv2D(
            filters=6, kernel_size=(5, 5),
            strides=(1, 1), padding='same',
            data_format='channels_last',
            name='conv_1', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), name='pool_1'),
        layers.Conv2D(
            filters=16, kernel_size=(5, 5),
            strides=(1, 1), padding='same',
            name='conv_2', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), name='pool_2'),
        layers.Flatten(),
        layers.Dense(
            units = 120, name='fc_1',
            activation='relu'),
        layers.Dense(
            units = 84, name='fc_2',
            activation='relu'),
        layers.Dropout(rate=0.5),
        layers.Dense(
            units=10, name='output',
            activation='softmax')
    ])

    model.build(input_shape=(None, 28, 28, 1))
    #print(model.layers[0].input_shape)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    history = model.fit(
        train, epochs=NUM_EPOCHS,
        validation_data=val,
        shuffle=True
    )

    Show_Data(history)

    model.save('number_recog_model.keras')

def Show_Data(history):
    ''' Visualize Results '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(NUM_EPOCHS)

    # training accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # loss measurements
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(fontsize=15)
    plt.title('Training and Validation Accuracy')
    plt.show()

def Test_Data(test):
    model = Load_Model()
    x = [item[0] for item in test]
    y = [item[1] for item in test]
    tensor_x = tf.convert_to_tensor(x)
    tensor_y = tf.convert_to_tensor(y)
    
    test_loss, test_acc = model.evaluate(
        x=tensor_x, y=tensor_y, batch_size=BATCH_SIZE, verbose=1)
    print('Test loss:', test_loss)
    print('Test Accuracy:', test_acc)


#Train_Model(train, val)

Test_Data(test)