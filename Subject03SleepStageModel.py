import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from keras.models import Sequential
from keras.layers import GaussianNoise, GlobalAveragePooling1D, InputLayer,Reshape, LSTM, Bidirectional, Dropout, Dense, GRU, TimeDistributed, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Activation
from keras import regularizers
from keras.initializers import glorot_normal
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras import backend as K

num_features = 43
label_idx = 44
time_window = 3
window_label_idx = round(time_window / 2)

def f1(y_true, y_pred):
    #f1=f1_score(y_true, y_pred, average=None)
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp/(tp + fp + K.epsilon())
    r = tp/(tp + fn + K.epsilon())

    f1 = 2*p*r/(p+r+K.epsilon())
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def getModel():
    model = Sequential()
   
    model.add(Conv1D(64, 4, strides=1, activation=None, padding='causal',
                              bias_initializer='zeros', kernel_initializer=glorot_normal(seed=201900),
                              kernel_regularizer=regularizers.l2(0.01), input_shape=(time_window,num_features)))
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
    model.compile(loss='categorical_crossentropy', metrics=[f1],
              optimizer='adam')

    return model

def window(a, w = time_window, o = 1, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

def getData(train_files):
    totalData = np.empty((0,time_window,num_features+2))
    for f in train_files:
        arr = np.genfromtxt(f,delimiter=',')
        # get rid of csv labels, and change all sleep stages to sleep or no sleep
        arr = arr[1:len(arr),1:len(arr[0])]
        arr[np.argwhere(arr[0:len(arr),label_idx]==2),label_idx] = 1
        arr[np.argwhere(arr[0:len(arr),label_idx]==3),label_idx] = 1
        arr[np.argwhere(arr[0:len(arr),label_idx]==4),label_idx] = 1
        # normalize all rows for training
        arr[:,0:num_features] = tf.keras.utils.normalize(arr[:,0:num_features], axis=-1, order = 2)
        tempData = np.empty((len(arr[:,0])-time_window+1,time_window,num_features+2), float)
        for i in range(num_features+2):
            tempData[:,:,i] = window(arr[:,i],time_window)
        totalData = np.append(totalData, tempData, axis=0)
    # get labels for the data
    labels = totalData[:,window_label_idx,label_idx]
    labels = labels.astype(int)
    features = totalData[:,:,:num_features]

    return features, labels
    
files = glob.glob('./data/*.csv')
num_files = 14
tr, te = train_test_split(files[:num_files], test_size=0.3, random_state=49)
kf = KFold(n_splits=5)
model = getModel()

round = 0
for train, test in kf.split(tr):
    round += 1
    train_files = [files[i] for i in train]
    test_files = [files[i] for i in test]
    
    train_data, train_label = getData(train_files)
    test_data, test_label = getData(test_files)
    train_label=to_categorical(train_label,2)
    #test_label=to_categorical(test_label,2)
    
    weight_path = './weiights/weight_best_'+str(round)+'.hdf5'
    checkpoint = ModelCheckpoint(weight_path, monitor='f1',
                                verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='f1', patience=200, mode='max')
    csv_logger = CSVLogger('./weiights/weight_training_'+str(round)+'.log')
    callbacks_list = [checkpoint, early_stopping, csv_logger]

    model.fit(train_data, train_label, shuffle=False, epochs=200, batch_size=4000, callbacks=callbacks_list)
    model.load_weights(weight_path)
    model.save('./model_'+str(round)+'fold')

    prediction = model.predict(test_data, batch_size=4000, verbose=1)
    predicted_labels = np.argmax(prediction, axis=1)
    cm = confusion_matrix(test_label, predicted_labels)
    total_samples = np.sum(cm)
    cm_prob = cm.astype('float') / total_samples
    print(cm_prob)

