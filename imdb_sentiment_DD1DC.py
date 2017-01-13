'''
Adapted from Keras example.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import os
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Merge, merge, BatchNormalization
from keras.layers import Embedding
from keras.layers import Convolution1D, AtrousConvolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K


os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "1"


# set parameters:
max_features = 1000
maxlen = 400
batch_size = 32
nb_filter = 150
filter_length = 3
hidden_dims = 250
nb_epoch = 10

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)



def word_index_to_one_hot(X):
    number_of_sequences = X.shape[0]
    X_one_hot = np.zeros((number_of_sequences, maxlen, max_features), dtype=np.bool)
    for sequence_number in range(number_of_sequences):
        for sequence_position, word_index in enumerate(X[sequence_number,:]):
            X_one_hot[sequence_number, sequence_position, word_index] = np.True_
    return X_one_hot


print('Build model...')

input = Input(shape=(maxlen, max_features))

"""
input = Input(shape=(maxlen,))


embedded_sequence = Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2)(input)
"""

ac1 = AtrousConvolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='relu',
                        atrous_rate=1,
                        subsample_length=1)(input)

ac1_merged = merge([input, ac1], mode='concat', concat_axis=2)
ac1_merged = BatchNormalization()(ac1_merged)

ac2 = AtrousConvolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='relu',
                        atrous_rate=2,
                        subsample_length=1)(ac1_merged)
ac2 = Dropout(0.1)(ac2)
ac2_merged = merge([input, ac1, ac2], mode='concat', concat_axis=2)
ac2_merged = BatchNormalization()(ac2_merged)

ac3 = AtrousConvolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='relu',
                        atrous_rate=4,
                        subsample_length=1)(ac2_merged)
ac3 = Dropout(0.1)(ac3)
ac3_merged = merge([input, ac1, ac2, ac3], mode='concat', concat_axis=2)
ac3_merged = BatchNormalization()(ac3_merged)

ac4 = AtrousConvolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='relu',
                        atrous_rate=8,
                        subsample_length=1)(ac3_merged)
ac4 = Dropout(0.1)(ac4)
ac4_merged = merge([input, ac1, ac2, ac3, ac4], mode='concat', concat_axis=2)
ac4_merged = BatchNormalization()(ac4_merged)

# we use max pooling:
x = GlobalMaxPooling1D()(ac4_merged)

# We add a vanilla hidden layer:
x = Dense(hidden_dims)(x)
x = Dropout(0.2)(x)
x = Activation('relu')(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
x = Dense(1)(x)
output = Activation('sigmoid')(x)

model = Model(input=input, output=output)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(word_index_to_one_hot(X_train), y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(word_index_to_one_hot(X_test), y_test))
