import os
import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import psutil
from keras.applications import MobileNet
import pandas as pd
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
import tensorflow as tf
import random as rn

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sd = 1 # Here sd means seed.
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(sd)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)

HEIGHT = 128
WIDTH = 128
bs = 64
p = psutil.Process(os.getpid())

try:

    p.nice(0)  # set>>> p.nice()10

except:

    p.nice(psutil.HIGH_PRIORITY_CLASS)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = os.getcwd() + '/dataset/'

df=pd.read_csv('celeba.csv')

columns = [['Eyeglasses', 'Male', 'Mustache', 'No_Beard', 'Smiling', 'Wearing_Hat', 'Young'], ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair','Gray_Hair']]

np.random.seed(0)
msk = np.random.rand(len(df)) < 0.975

valid = df[~msk]
train = df[msk]

valid_hair = valid[valid[columns[1]].sum(axis=1) == 1]
train_hair = train[train[columns[1]].sum(axis=1) == 1]

def network(cols, loss_f, activation_f, train_gen, valid_gen, l):

    base_model=MobileNet(weights='imagenet', input_shape=(HEIGHT, WIDTH, 3), include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
    x=base_model.layers[l].output
    x=GlobalMaxPooling2D()(x)
    preds=Dense(len(cols),activation=activation_f)(x)

    model=Model(inputs=base_model.input,outputs=preds)

    for layer in model.layers:
        layer.trainable=True

    opt = Adam(lr=0.001)

    model.compile(loss=loss_f, optimizer=opt, metrics=['acc', BinaryAccuracy(), Precision(), Recall()])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint(str(l)+'_'.join(cols).lower()+'_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    dc = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-9)

    model.save_weights(str(l)+'_'.join(cols).lower()+'_best_model.h5')
    model_json = model.to_json()

    with open(str(l)+'_'.join(cols).lower()+'_model.json', 'w') as json_file:
        json_file.write(model_json)

    datagen=ImageDataGenerator(rescale=1./255)

    train_generator=datagen.flow_from_dataframe(
        dataframe=train_gen,
        directory=path,
        x_col='Filename',
        y_col=cols,
        batch_size=bs,
        color_mode='rgb',
        class_mode='raw',
        target_size=(HEIGHT,WIDTH))

    valid_generator=datagen.flow_from_dataframe(
        dataframe=valid_gen,
        directory=path,
        x_col='Filename',
        y_col=cols,
        batch_size=bs,
        color_mode='rgb',
        class_mode='raw',
        target_size=(HEIGHT,WIDTH))

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    h = model.fit_generator(generator=train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=valid_generator,
            validation_steps=STEP_SIZE_VALID,
            epochs=50,
            verbose = 1,
            callbacks=[es, mc, dc],
            workers = 10,
            )

    return model, h.history

if __name__ == '__main__':

    model1, h1 = network(columns[0], 'binary_crossentropy', 'sigmoid', train, valid, 54)
    model2, h2 = network(columns[1], 'categorical_crossentropy', 'softmax', train_hair, valid_hair, 54)









