import numpy as np
import pickle
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.model_selection import train_test_split as split
from PIL import Image

# вспомогательная функция для загрузки всего датасета
def load_cifar10():
    files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
    metafile = "batches.meta"
    databatches = []
    labels = []
    # прогружаем данные
    for fn in files:
        with open(fn, 'rb') as fo:
            dct = pickle.load(fo, encoding='bytes')
            databatches.append(dct[b'data'])
            labels += dct[b'labels']
    # меняем форму массива, для отправки в нейронку
    data = np.concatenate(databatches).reshape((-1, 3, 32, 32))
    data = np.moveaxis(data, 1, -1)
    # подгружаем лейблы
    with open(metafile, 'rb') as fo:
        mdct = pickle.load(fo, encoding='bytes')
        labelnames = mdct[b'label_names']
    labels = np_utils.to_categorical(labels, len(labelnames))
    return (data, labels, labelnames)

# подгрузить классы
def get_classes():
    metafile = "batches.meta"
    with open(metafile, 'rb') as fo:
        mdct = pickle.load(fo, encoding='bytes')
        labelnames = mdct[b'label_names']
    return labelnames

# вспомогательная функция, создающая модель по списку слоев
def create_model(layers, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    """
    Create and compile a Keras model using given layers.
    """
    model = Sequential()
    for layer in layers:
        if len(layer) == 1:
            model.add(eval(layer[0])())
        elif len(layer) == 2:
            model.add(eval(layer[0])(**layer[1]))
        else:
            raise ValueError("Wrong layer specification")
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

# функция, возвращающая вероятности для каждого класса
picture_shape = (32, 32)
def predict_probs(model, filename):
    img = Image.open(filename)
    newimg = img.resize(picture_shape)
    imdata = np.array(newimg)
    return model.predict(imdata[np.newaxis,:])[0]

# натренировать и сохранить веса
def train_save(epochs=30, jsonfile="model.json", weightfile="model.h5"):
    X, y, classes = load_cifar10()
    nclasses = len(classes)
    # слои моей модели
    layers = [("Conv2D", {"filters":32, "kernel_size":(3,3), "strides":(1,1), "activation":"relu", "input_shape":X.shape[1:], "padding":"same"}),
              ("BatchNormalization", {}),
              ("Conv2D", {"filters":32, "kernel_size":(3,3), "strides":(1,1), "activation":"relu", "padding":"same"}),
              ("MaxPooling2D", {"pool_size":(2,2), "padding":"same"}),
              ("BatchNormalization", {}),
              ("Dropout", {"rate":0.5}),
              ("Conv2D", {"filters":64, "kernel_size":(3,3), "strides":(1,1), "activation":"relu", "padding":"same"}),
              ("BatchNormalization", {}),
              ("Conv2D", {"filters":64, "kernel_size":(3,3), "strides":(1,1), "activation":"relu", "padding":"same"}),
              ("BatchNormalization", {}),
              ("MaxPooling2D", {"pool_size":(2,2), "padding":"same"}),
              ("Dropout", {"rate":0.5}),
              ("Flatten", {}),
              ("Dense", {"units":64, "activation":"relu"}),
              ("Dropout", {"rate":0.5}),
              ("Dense", {"units":nclasses, "activation":"softmax"})]
    # тренируем
    model = create_model(layers)
    Xtrain, Xtest, ytrain, ytest = split(X, y, test_size=0.2)
    model.fit(Xtrain, ytrain, batch_size=256, epochs=epochs, validation_data = (Xtest, ytest))
    # сохранить модель
    model_json = model.to_json()
    with open(jsonfile, "w") as json_file:
        json_file.write(model_json)
    # сохранить веса (БОЛЬШОЙ ФАЙЛ)
    model.save_weights(weightfile)
    return model

# восстановить модель с диска
def restore_model(jsonfile="model.json", weightfile="model.h5", loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
    # загрузить модель
    jsonfl = open(jsonfile, 'r')
    loaded_model_json = jsonfl.read()
    jsonfl.close()
    loaded_model = model_from_json(loaded_model_json)
    # загрузить веса
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return loaded_model


if __name__ == '__main__':
    # запусти это первый раз, потом закомментируй (тренировка и сохранение модели)
    model = train_save(epochs=30)
    # если модель уже натренирована и сохранена, закомментируй предыдущую строчку и раскомментируй эту
    #model = restore_model()
    # пример распознавания картинки
    res = predict_probs(model, "a380.jpg")
    classes = get_classes()
    print({classes[i]:res[i] for i in range(len(res))})
