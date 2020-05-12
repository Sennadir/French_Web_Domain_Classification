"""
The current module is a pre-processing module, which will take the raw text
and will extract tokens and delete different stopwords and punctuations.
It will generate as an output a dictionnary of the vocabulary with the different
occurences.

__version__ = 1.0

"""


import codecs
from os import path
import pandas as pd
import numpy as np
import networkx as nx
import re

import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import fasttext as ft

from collections import Counter
import pickle


nlp = spacy.load('fr_core_news_md')

lid_model = ft.load_model("lid.176.ftz")

french_stopwords = stopwords.words('french')
other = ['être', 'avoir', 'tout', 'site', 'recherche', 'français',\
        'rechercher', 'accueil', 'plus', 'faire']
french_stopwords += other

# Occurence Parameters
min_occurences = 20
min_characters = 2
max_characters = 12

# Data Paths
data_path = 'pickles/'
files_path = 'text/text/'

## ------------------------------ File Processing --------------------------- ##


def process_file(file):
    """
    Function to process a single file.

    ---
    Parameters:
        file : str
                File name in the Data Folder.

    ---
    Results:
        my_data : dict
                Dictionnary containing the number of occurences of each word in
                the document.
    """

    try:
        with open(files_path+data_path+file+'.pickle', 'rb') as handle:
            my_data = pickle.load(handle)

    except:
        f = open(files_path + file, "r", encoding="utf-8")
        l = []
        try:
            for x in f:
                #removing all non-alphanumeric characters except apostrophes
                x = re.sub(r'[^\w'+"'’ "+']', "",x)
                x = x.lower()
                x = x.replace('\n', "")
                x = re.sub(' +', ' ', x) # removing duplicate spaces
                x = x.strip()
                # checking if line is not empty and is in french
                if (x != '' and lid_model.predict(x)[0][0]=='__label__fr'):
                    # lemmatization using spacy
                    doc = nlp(x)
                    tokens = []
                    for token in doc:
                        tokens.append(token.lemma_)
                    x = ' '.join(tokens)
                    l.append(x)
        except:
            pass
        my_data = {}
        for lines in l :
            my_elements = lines.split(' ')

            for element in my_elements:
                if element.isalpha():
                    # removing stopwords
                    if not element in french_stopwords:
                        # checking if entry exists in file vocabulary
                        if element in my_data.keys():
                            my_data[element] += 1

                        else:
                            my_data[element] = 1


        my_data = dict(filter(lambda elem: \
        lid_model.predict(elem[0])[0][0]=='__label__fr', \
                                                        my_data.items()))
        my_data = dict(filter(lambda elem: \
        (len(elem[0]) > min_characters) and (len(elem[0]) < max_characters), \
                                                        my_data.items()))
        with open(files_path+data_path+file+'.pickle', 'wb') as handle:
            pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return my_data

## ------------------------------ Vocab Extract ----------------------------- ##


def get_vocab(files=[], pkl = True):
    """
    Function to extract the vocabulary contained in the train data

    ---
    Parameters:
        train_data : DataFrame Pandas
                Training DataFrame containg the names of the training files
                to be used

    ---
    Results:
        my_data : dict
                Dictionnary containing the number of occurences of each word in
                all the documents
    """

    if not pkl:
        vocab = {}
        for file in tqdm(files):

            vocab_file = process_file(str(file))
            vocab = dict(Counter(vocab) + Counter(vocab_file))

        vocab = dict(filter(lambda elem: elem[1] >= min_occurences, \
                                                    vocab.items()))
        print(len(vocab))
        with open('vocab.pickle', 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        vocab = {}
        try:
            with open('vocab.pickle', 'rb') as handle:
                vocab = pickle.load(handle)
        except:
            print('no vocabulary found !')
    vocab = {k: v for k, v in sorted(vocab.items(), \
                                        reverse=True,key=lambda item: item[1])}

    for key in other:
        vocab.pop(key, None)
    return vocab


if __name__ == '__main__':
    # Data Paths
    data_path = '../Data/'
    files_path = '../Data/text/text/'

    # Train Data
    train_data = pd.read_csv(data_path + 'train.csv', header = None)
    train_data.columns = ['File', 'Type']

    for files in train_data['File']:
        file_vocab = process_file(file=str(files))

    for files in test_data['File']:
        file_vocab = process_file(file=str(files))

    vocab = get_vocab()
