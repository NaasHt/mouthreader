import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout, BatchNormalization, Input,
                                     ZeroPadding2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

face_shape = (48, 48, 1)
batch_size = 64
num_classes = 6
EPOCHS_FACE = 50

def build_face_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

face_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

face_train = face_datagen.flow_from_directory(
    './FER-2013/train', target_size=(48, 48), color_mode='grayscale',
    batch_size=batch_size, class_mode='categorical', subset='training')

face_val = face_datagen.flow_from_directory(
    './FER-2013/train', target_size=(48, 48), color_mode='grayscale',
    batch_size=batch_size, class_mode='categorical', subset='validation')

face_model = build_face_model(face_shape)
face_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

face_model.fit(face_train, validation_data=face_val, epochs=EPOCHS_FACE)
face_model.save('face_model.h5')
