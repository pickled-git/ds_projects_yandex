from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D
import numpy as np

def load_train(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255,)

    train_gen_flow = datagen.flow_from_directory(path, target_size=(150, 150), 
                                            batch_size=16, 
                                            class_mode='sparse', 
                                            seed=12345)

    return train_gen_flow

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=6,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=12, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=5,
               steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)
        
    train_datagen_flow = train_data
    test_datagen_flow = test_data

    model.fit(train_datagen_flow,
              validation_data=test_datagen_flow,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model
