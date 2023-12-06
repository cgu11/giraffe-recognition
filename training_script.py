import tensorflow.keras as keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers

import sys

def load_data(class_mode:str='categorical'):
    train_datagen = ImageDataGenerator(
        # shear_range=10,
        # zoom_range=0.2,
       # horizontal_flip=True,
        preprocessing_function=preprocess_input,
        validation_split = 0.3)

    train_generator = train_datagen.flow_from_directory(
        'final_training/',
        batch_size=2,
        class_mode=class_mode,
        subset='training',
        shuffle=True,
        target_size=(224,224))
    print(train_generator.class_indices)
    validation_generator = train_datagen.flow_from_directory(batch_size=2,
                                                 directory='final_training',
                                                 shuffle=True,
                                                 target_size=(224, 224), 
                                                 subset="validation",
                                                 class_mode=class_mode)


    
    return train_generator, validation_generator

def create_network(n_classes):
    network_base = ResNet50(include_top=False, weights='imagenet')

    for layer in network_base.layers:
        layer.trainable = False

    x = network_base.output
    x = layers.GlobalAveragePooling2D(name="GiraffePooling")(x)
    x = layers.Dense(128, activation='relu', name="GiraffeDense")(x)
    predictions = layers.Dense(n_classes, activation='softmax', name="GiraffeOut")(x)
    model = Model(network_base.input, predictions)

    optimizer = keras.optimizers.legacy.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    
    return model

def train(model, train_gen, test_gen):
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=test_gen
    )
    return history
# train_gen = load_data()
if __name__=="__main__":
    model_iteration = sys.argv[1]
    train_gen, test_gen = load_data()
    model = create_network(2)

    history = train(model, train_gen, test_gen)

    # architecture and weights to HDF5
    model.save(f'models/model{model_iteration}.h5')