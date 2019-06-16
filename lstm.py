"""Train and test LSTM classifier"""
# import dga_classifier.data as data
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical
import sklearn
from sklearn.model_selection import train_test_split

map_dict = {'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17,
            'h': 18, 'i': 19, 'j': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24,
            'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 'u': 31,
            'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36, '.': 37, '-': 38,
            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10}

max_epoch=25
max_len = 65

def sentences_to_indices(X, max_len):

    m = X.shape[0]
    x_indices = np.zeros((m,max_len))
    for i in range(m):
        j = 0
        for w in X[i]:
            x_indices[i,j] = map_dict[w]
            j += 1
    return x_indices


def build_model(max_features, maxlen ,class_number):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(class_number))
    model.add(Activation('softmax'))

    return model

def run(X,labels,class_number,max_features,batch_size=128,maxlen=65):
    model = build_model(max_features,maxlen,class_number)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.1,random_state=6)

    for ep in range(max_epoch):
        model.fit(X_train, y_train,epochs=3,batch_size=batch_size)
        model.save('model/model_max_len=65_'+str(ep)+'.h5')
        loss, acc = model.evaluate(X_test, y_test)
        print(acc)

if __name__ =='__main__':
    indata = pd.read_csv('dga_domain.csv',low_memory=False)
    indata = np.array(indata)
    X = indata[:,1]
    labels = indata[:,0]
    # maxlen = np.max([len(x) for x in X])
    max_features = len(map_dict)+1
    x_indices = sentences_to_indices(X,max_len)
    one_hots_labels = to_categorical(labels)
    class_number = len(one_hots_labels[0])
    run(x_indices,one_hots_labels,class_number,max_features,max_len)
