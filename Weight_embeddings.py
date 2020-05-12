import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import pickle

from sklearn.model_selection import train_test_split
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.constraints import unit_norm

# Path to data
path_to_data = './text/'

# Parameters
Load_embegginds = True # Choose between loading or creating the vocab.
train_embedding_doc = False # Loading or training the embedding vectors

batch_size = 128
num_classes = 8
epochs = 100

def extract_file_embedding(file):
    """
    This function aims to applicate the Weight Embeddings method. The goal is
    to extract an embedding vector of a document (single example) from the
    embedding vectors of the word composing the document. The different vectors
    are weighted in function of the frequence of the word in the document.

    ---
    Parameters:
        file : str
                File name in the Data Folder.

    ---
    Output:
        vect_file : n_array
                The embedding vector that has been generated.
    """

    file = './embeds/Vocab_occurences/pickles/' + file
    with open(file + '.pickle', 'rb') as handle:
        my_data = pickle.load(handle)

    w_c = max(vocab.values()) / max(my_data.values())
    common_vocab = {k: float(w_c * my_data[k]/vocab[k]) \
                                                for k in my_data.keys() & vocab}

    embedding_words = {k: np.array(embeddings[k]) * common_vocab[k] \
                                    for k in common_vocab.keys() & embeddings}

    vect_file = sum(embedding_words.values())

    return vect_file



if __name__ == '__main__':

    # Load Vocabs
    train_data = pd.read_csv('./embeds/' + 'train_noduplicates.csv',\
                                header = None)
    train_data.columns = ['File', 'Type']

    test_data = pd.read_csv('./embeds/' + 'test.csv', header = None)
    test_data.columns = ['File']

    enc = OrdinalEncoder()
    X = train_data['Type']
    labels = enc.fit_transform(np.array(X).reshape(-1,1))
    train_data['Labels'] = labels


    if Load_embegginds == True:
      my_vectors = {}
      i = 0
      for line in open('./cc.fr.300.vec'):
          fields = line.strip().split(" ")
          nom = fields[0].lower()
          if nom.isalpha():
              my_vectors[nom] = [float(v) for v in fields[1:]]

    else:
      with open('./embeds/pickle_embed.pickle', 'rb') as handle:
        embeddings = pickle.load(handle)

    with open('./embeds/vocab_clean.pickle', 'rb') as handle:
        vocab = pickle.load(handle)


    # generate our embeddings
    if train_embedding_doc == True:
      from os import listdir
      test = []
      vocab_embedding_docs = {}
      i = 0

      for file in tqdm(listdir('./embeds/Vocab_occurences/pickles')):
        file = file.split('.')[0]
        try:
          if (len(extract_file_embedding(file))) == 300:
            vocab_embedding_docs[file] = extract_file_embedding(file)
        except:
          vocab_embedding_docs[file] = np.array([0 for t in range(300)])
          test.append(file)
          pass

    else:
      with open('./embeds/doc_vocab_embed.pickle', 'rb') as handle:
        vocab_embedding_docs = pickle.load(handle)


    # Let's generate our Features
    my_list = []
    X = []
    y = []
    for element in vocab_embedding_docs.keys():
      try:
        if len(vocab_embedding_docs[element]) == 300:
          y_t = train_data[train_data['File'] == int(element)]['Labels'].iloc[0]
          y.append(y_t)
          X.append(vocab_embedding_docs[element])
      except:
        my_list.append(element)

    X = np.vstack(X)
    y = np.array(y).reshape(-1, 1)


    # Let's use a MLP
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(300,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='tanh', kernel_constraint=unit_norm()))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', bias = True))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
