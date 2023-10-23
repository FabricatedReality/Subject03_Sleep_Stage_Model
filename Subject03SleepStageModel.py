import glob
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from keras.models import Sequential
from keras.layers import GaussianNoise, GlobalAveragePooling1D, InputLayer,Reshape, LSTM, Bidirectional, Dropout, Dense, GRU, TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Activation
from keras import optimizers, regularizers
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, EarlyStopping

def getModel():
    model = Sequential()
   
    model.add(Conv1D(64, 4, strides=1, activation=None, padding='causal',
                              bias_initializer='zeros', kernel_initializer=glorot_normal(seed=201900),
                              kernel_regularizer=regularizers.l2(0.01), input_shape=(20,43)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
   
    model.add(Conv1D(64, 4, strides=1, activation=None, padding='causal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
   
    model.add(Conv1D(64, 4, strides=1, activation=None, padding='causal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
   
    model.add(Conv1D(64, 4, strides=1, activation=None, padding='causal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
   
    model.add(LSTM(128, activation='tanh', return_sequences=False))
    model.add(GaussianNoise(0.1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam')

    return model

def getData(train_files):
    for f in train_files:
        arr = np.genfromtxt(f,delimiter=',')
        print(arr)

    
files = glob.glob('./data/*.csv')
kf = KFold(n_splits=3)
model = getModel()

for train, test in kf.split(files):
    train_files = [files[i] for i in train]
    test_files = [files[i] for i in test]
    data, label = getData(train_files)

    checkpoint = ModelCheckpoint('./', monitor='val_f1',
                                verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_f1', patience=200, mode='max')
    callbacks_list = [checkpoint, early_stopping]
    model.fit(data, label, shuffle=False, epochs=200, batch_size=4000, callbacks=callbacks_list)

    
