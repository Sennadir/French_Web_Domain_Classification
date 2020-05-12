"""
The current module is a the Bert Approach application to our example. It aims
to applicate the Bert (Attention with RNN) model and extract embeddings that
could be used later on.

__version__ = 1.0

"""

import codecs
import sys
import csv
from unidecode import unidecode
import pandas as pd
import numpy as np
import string
import networkx as nx
import scipy.sparse as sp
import nltk
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('french'))

from transformers import CamembertTokenizer, CamembertModel, \
                                            CamembertForSequenceClassification
import torch


# Path to Data
data_path = "./data/text/text/"
edgelist_path = "./data/edgelist.txt"

start_fraction = 0
end_fraction = 1

# Parameters
GPU = True # If you have access to a GPU(in our case, we are using Google Colab)


def build_graph():
    '''
    The following function aims to build a directed weighted graph from the
    edgelist.txt
    '''

    G = nx.read_weighted_edgelist(edgelist_path, create_using=nx.DiGraph())
    print("Number of nodes : ", G.number_of_nodes())
    print("Number of edges : ", G.number_of_edges())
    return G

def build_train_test(train_path, test_path):

    """
    The following function aims to read the train.csv and returns the train Ids
    and train labels and reads the test.csv and returns the test Ids
    """
    with open(train_path, 'r') as f:
        train_data = f.read().splitlines()

    train_hosts = list()
    y_train = list()
    for row in train_data:
        host, label = row.split(",")
        train_hosts.append(host)
        y_train.append(label.lower())

    df_train = pd.DataFrame(data= y_train, \
                        index = train_hosts, columns= ["class"]).reset_index()

    with open(test_path, 'r') as f:
        test_hosts = f.read().splitlines()
    df_test =  pd.DataFrame(data=[] ,\
                        index = test_hosts, columns= ["class"]).reset_index()
    return df_train, df_test

def write_submission(write_path, test_hosts, model_classes_list, predicted_probas):
    """
    The following function writes the submission file

    ----
    Parameters:
        - The path of the file to create
        - The test Ids (returned by build_train_test)
        - The classes labels as a list
        - The predicted probas for those class labels (same order)
    """
    with open(write_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        model_classes_list.insert(0, "Host")
        writer.writerow(model_classes_list)
        for i,test_host in enumerate(test_hosts):
            lst = predicted_probas[i,:].tolist()
            lst.insert(0, test_host)
            writer.writerow(lst)

def text_from_id(id):
    """
    Function to extract the text given a certain Id.

    ---
    Parameters:
        id :
                The Id for which we want to extract the text.

    ---
    Output:
        text : str
                The text corresponding the entred Id.
    """

    id = str(id)
    try :
        with codecs.open(data_path+id, 'r', encoding="utf-8") as f:
            text = f.readlines()
    except:
        with codecs.open(data_path+id, 'r', encoding="latin-1") as f:
            text = f.readlines()
    return text



def build_local_test(train_hosts, y_train, size_local_test=.25):
    """
    A simple split of the Data into a train and test set which will be used
    later on during the training and validation part.

    ---
    Parameters:
        train_hosts :
                Our Features matrix.
        y_train :
                Labels of corresponding to each record in our Features Matrix

        size_local_test :
                The ratio of split Train/Test


    """


    local_train, local_test, \
    local_y_train, local_y_test = train_test_split(train_hosts, y_train, \
                                    stratify=y_train,test_size=size_local_test)

    return local_train, local_y_train, local_test, local_y_test

def loglikelihood_score(predictions, y_true, classes_order):

    """
    The following function aims to compute our loss function that will be used
    during this project.

    ---
    Parameters:
        predictions :
                The predictions that were computed by our model
        y_true :
                The true labels

    ---
    Output:
        loss:
            The computed loss on the input predictions

    """


    dico = {v:k for k, v in enumerate(classes_order)}
    print(dico)
    loss = 0
    for i, cla in enumerate(y_true) :
        loss -= np.log(predictions[i, dico[cla]])
    loss = loss/len(y_true)
    return loss

if __name__ == '__main__':
    # Path to our train data
    train_path = "data/train_noduplicates.csv"
    test_path = "data/test.csv"



    train_data, test_data = build_train_test(train_path, test_path)


    # To avoid generating the test and train set each time, we will save it CSV
    try :
        train_data = pd.read_csv('train_data_with_text.csv')
        test_data = pd.read_csv('test_data_with_text.csv')

    except:
        train_data['text'] = train_data["index"].apply(text_from_id)
        train_data["class_codes"] = pd.Categorical(train_data["class"]).codes
        train_data.to_csv('train_data_with_text.csv')

        test_data['text'] = test_data["index"].apply(text_from_id)
        test_data["class_codes"] = pd.Categorical(test_data["class"]).codes
        test_data.to_csv('test_data_with_text.csv')


    train_data["text_processed"] = \
                                    train_data.text.apply(process_text, \
                                    args=(start_fraction, end_fraction,))
    train_data["text_processed"] = \
                                train_data.text_processed.apply(join_with_SEP)
    train_data["text_processed"] = \
                        replace_by_special_token(train_data["text_processed"])
    train_data["text_processed"] = \
                            punctuation_by_space(train_data["text_processed"])
    train_data["text_processed"] = \
                            train_data.text_processed.apply(remove_stop_words)
    train_data["text_processed"] = \
                                train_data.text_processed.apply(split_by_SEP)
    train_data["text_processed"] = \
                            train_data.text_processed.apply(remove_empty_rows)
    train_data["text_processed"] = \
                    train_data.text_processed.apply(remove_single_characters)
    train_data["text_processed_no_single_words"] = \
                        train_data.text_processed.apply(remove_single_word_rows)


    ## Transformers
    tokenizer_ = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertModel.from_pretrained('camembert-base')


    if GPU :
        model.eval();
        model.to('cuda');
    else:
        model.eval()

    # Generate the embeddings
    length = train_data.shape[0]
    for j in range(train_data.shape[0]):
        sys.stdout.write('\r'+str(j)+"/"+str(length))
        target = train_preprocessed["class_codes"].iloc[j]

        txt = ". ".join(train_data.text.iloc[j])
        try :
          tokens = tokenizer_.encode(txt, add_special_tokens=True)
          shape = len(tokens[1:-1])
          new_tokens = []
          for i in range(int(shape/510)+1):
              min_ = min((i+1)*510,shape)
              if min_ == shape :
                  L = [tokenizer_.cls_token_id] + tokens[i*510:min_] + \
                                                    [tokenizer_.eos_token_id]
                  new_tokens.append(L + [tokenizer_.pad_token_id]*(512 - len(L)))
              else :
                  new_tokens.append([tokenizer_.cls_token_id] + \
                                tokens[i*510:min_] + [tokenizer_.eos_token_id] )
          with torch.no_grad() :
              new_train_ = model(torch.tensor(new_tokens).cuda())[0][:,0,:]
          del new_tokens
          torch.cuda.empty_cache()
          if j == 0 :
            new_train = \
                    new_train_.detach().cpu().numpy().mean(axis=0).reshape(1,-1)
            new_train_target = [target]
          else :
            new_train = np.concatenate((new_train,
                new_train_.detach().cpu().numpy().mean(axis=0).reshape(1,-1)),
                                       axis=0)
            new_train_target.append(target)
        except :
          new_train = np.concatenate((new_train, np.zeros((1,768))), axis=0)
          new_train_target.extend([target])
    new_train = np.array(new_train)

    # Create a DataFrame containing the Embeddings and the Labels
    train_camembert = pd.DataFrame(new_train)
    train_camembert['target'] = new_train_target

    X_train = train_camembert.iloc[:,:-1]
    y_train = train_camembert.target

    X_train = X_train.values
    y_train = y_train.values

    # A grid Search to tune the model
    grid={"C":np.logspace(-1,3, num = 30)}

    # We will use a Classical LogistricRegression with a GridSearch evaluated
    # on a 3 Folders Cross Validation

    logreg = LogisticRegression(solver='lbfgs',\
                                multi_class='auto', max_iter=25000, n_jobs=-1)

    classes_order = LogisticRegression(solver='lbfgs', \
                    multi_class='auto').fit(X_train[:, :2], y_train).classes_
    score_function = make_scorer(loglikelihood_score,\
        greater_is_better=False, classes_order=classes_order, needs_proba=True)

    logreg_cv = GridSearchCV(logreg,grid,cv=3,\
                                verbose=3, n_jobs=-1, scoring=score_function)


    logreg_cv.fit(X_train, y_train)

    print(logreg_cv.best_params_)
    print('Score on the local test : ', logreg_cv.best_score_)

    # Let's try a more robust model

    # Create the parameter grid based on the results of random search
    param_grid = {
        'max_depth': [20, 40, 60],
        'max_features': [2, 5, 10, 15, 20],
        'n_estimators': [100, 200, 300, 1000]
    }

    X_train_1, X_test_1, Y_train_1, Y_train_2 = \
                            train_test_split(X_train, y_train, test_size = 0.2)


    rf = RandomForestClassifier()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,\
                        scoring=score_function,cv = 3, n_jobs = -1, verbose = 3)

    grid_search.fit(X_train_1, Y_train_1)

    print(grid_search.best_params_)
    print('Score of Grid Search : ', grid_search.best_score_)
    print('Score on test', grid_search.score(X_test_1, Y_train_2) )
