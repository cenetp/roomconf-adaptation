import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

import generate_cnn_data as gcd

batch_size = 512
num_classes = 2
epochs = 200
max_room_count = gcd.max_room_count
convnet_type = 'discriminator'
dirname = 'data_flp'

# reshape training and test data to connection maps with [max_room_count x max_room_count] size
x_train = np.array(gcd.dataset(num_classes, convnet_type, 'train', max_room_count, '')) \
    .reshape(10000, max_room_count, max_room_count, 1)
x_test = np.array(gcd.dataset(num_classes, convnet_type, 'test', max_room_count, '')) \
    .reshape(2000, max_room_count, max_room_count, 1)

# converts this into a matrix with as many columns as there are classes. The number of rows stays the same
# @see https://keras.io/utils/
y_train = keras.utils.to_categorical(np.array(gcd.classes(num_classes, convnet_type, 'train')), num_classes)
y_test = keras.utils.to_categorical(np.array(gcd.classes(num_classes, convnet_type, 'test')), num_classes)

# define model for discriminator
discriminator_model = Sequential()
discriminator_model \
    .add(Conv2D(32, kernel_size=(3, 3), activation='elu', input_shape=(max_room_count, max_room_count, 1)))
discriminator_model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
discriminator_model.add(MaxPooling2D(pool_size=(2, 2)))
discriminator_model.add(Dropout(0.25))
discriminator_model.add(Flatten())
discriminator_model.add(Dense(128, activation='elu'))
discriminator_model.add(Dropout(0.5))
discriminator_model.add(Dense(num_classes, activation='softmax'))

discriminator_model.compile(
    loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Nadam(), metrics=['accuracy'])

discriminator_model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
test_eval = discriminator_model.evaluate(x_test, y_test, verbose=0)
print('loss:', test_eval[0], ', accuracy:', test_eval[1])

discriminator_model.save(dirname + '/' + convnet_type + '/saves/discriminator.h5')
