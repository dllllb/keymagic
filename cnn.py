from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, RepeatVector, TimeDistributed
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import RMSprop

import common

def model_cnn_lstm(props):
    n_points = props['n_points']
    n_chars = props['n_chars']
    max_letters = props['max_letters']
    
    model = Sequential()
    model.add(Convolution1D(8, 4, input_shape=(n_points, 3)))
    model.add(Activation('relu'))
    model.add(Convolution1D(8, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Convolution1D(1, 1))
    model.add(Flatten())
    model.add(RepeatVector(max_letters))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(n_chars)))
    model.add(TimeDistributed(Activation('softmax')))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=props['lr']))
    return model


def init_props(overrides):
    defaults = {
        'train_data': 'tiny-shakespeare.txt',
        'n_points': 100,
        'n_chars': 27,
        'max_letters': 20,
        'lr': .01,
        'nb_epoch': 10,
        'batch_size': 256,
    }
    return {**defaults, **overrides}


def fit_model_cnn_lstm(props, char_limit=None):
    props = init_props(props)
    
    kbrd = common.keyboardIOS7()
    
    X, y = common.generate_dataset_words(props, kbrd, char_limit)
    
    model = model_cnn_lstm(props)
    
    bs = props['batch_size']
    nb_epoch = props['nb_epoch']
    
    hist = model.fit(X, y, batch_size=bs, epochs=nb_epoch)
    
    return model, hist


def test_cnn_lstm():
    props = {
        'nb_epoch': 1,
        'batch_size': 64,
    }
    
    model, hist = fit_model_cnn_lstm(props, char_limit=1000)
    
    kbrd = common.keyboardIOS7()
    l = common.generate_input('man', 100, kbrd)
    pred = common.predict(model, kbrd, l)
    print(pred)
