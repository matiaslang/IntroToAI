#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap

def plotTrainingProgress(model, modelname, epochs):
    """
    Tässä funktiossa tallennetaan kuvaajaan opetusjoukon ja validointijoukon luokittelutarkkuudet jokaisen opetusjakson jälkeen
    """
    plt.figure(figsize=(10,10))
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    train_acc = model.history['acc'][-1]
    val_acc = model.history['val_acc'][-1]
    print('\n\nTraining accuracy after {} epoch: {}'.format(epochs, round(train_acc,3)))
    print('Validation accuracy after {} epoch: {}'.format(epochs, round(val_acc,3)))
    plt.title("Training and validation accuracies for {}".format(modelname))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.figtext(0.02,0.02,"Training accuracy after {} epoch: {}\nValidation accuracy after {} epoch: {}".format(epochs, round(train_acc,3), epochs, round(val_acc,3)))
    plt.xticks(np.arange(len(model.history['acc'])), np.arange(len(model.history['acc']))+1)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    filename = modelname + "_accuracies.png"
    plt.savefig(filename)
	
def plotImage(i, predictions_array, true_label, img, class_names):
    """
    Tässä funktiossa tulostetaan kuvaajaan validointijoukon kuva, kuvan oikea luokka, ennustettu luokka ja ennustuksen varmuus
    """
    predictions_array, true_label, img = predictions_array[i], *true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([]) 
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

def plotValueArray(i, predictions_array, true_label):
    """
    Tässä funktiossa tulostetaan pylväsdiagrammiin, millä varmuudella luokittelijan mukaan validointijoukon kuva on luokiteltu eri luokkiin
    """
    predictions_array, true_label = predictions_array[i], *true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
		
def model1():
    """
    Tässä funktiossa on toteutettu mallin 1 konvoluutioneuroverkon rakenne
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
	
def model2():
    """
    Tässä funktiossa on toteutettu mallin 2 konvoluutioneuroverkon rakenne
    """
    #-------TÄHÄN SINUN KOODI--------  
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3 ,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
    #--------------------------------

parser = ap.ArgumentParser()
parser.add_argument("-m", "--model", help="Choose either model1 or model2", required="True")
args = vars(parser.parse_args())
modelname = str(args["model"])
batch_size = 32
num_classes = 10
epochs = 10

# Jaetaan data opetusjoukkoon ja validointijoukkoon
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer','Dog', 'Frog', 'Horse', 'Ship', 'Truck']

if modelname == "model2":
    model = model2()
else:
    modelname = "model1"
    model = model1()

# Alustetaan RMSprop optimoija
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Opetetaan malli käyttämällä RMSprop optimoijaa
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255

output = model.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val_cat), shuffle=True)

# Tulostetaan opetusjoukon ja validointijoukon luokittelutarkkuudet yhden jakson välein kuvaajaan
plotTrainingProgress(output, modelname, epochs)

# Ennustetaan validointijoukon 15 ensimmäiselle näytteelle luokat ja tulostetaan tämä uuteen kuvaajaan
y_pred = model.predict(x_val)
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plotImage(i, y_pred, y_val, x_val, class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plotValueArray(i, y_pred, y_val)
fig = plt.gcf()
fig.suptitle("Classification results for 15 first samples of the validation set for {}".format(modelname))
plt.figtext(0.02,0.02,"The class names in the right order: {}".format(", ".join(class_names)))
filename = modelname + "_predictions.png"
plt.savefig(filename)
plt.show()
