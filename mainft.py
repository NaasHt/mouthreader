import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mouth_shape = (17, 48, 1)
batch_size = 32
EPOCHS = 30

base_model = load_model('face_model.h5')
for layer in base_model.layers[:-5]:
    layer.trainable = False

mouth_input = Input(shape=mouth_shape)
x = ZeroPadding2D(padding=((15, 16), (0, 0)))(mouth_input)
output = base_model(x)
model = Model(inputs=mouth_input, outputs=output)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_gen = datagen.flow_from_directory(
    './cropped_mouth',
    target_size=(17, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    './cropped_mouth',
    target_size=(17, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save('fine_tuned_mouth_model.h5')
