import tensorflow as tf
import pkg_resources
import importlib
importlib.reload(pkg_resources)
import os
import numpy as np
from warnings import simplefilter
from tensorflow.keras import Model, Input, backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from transformers import TFBartModel, BartConfig, BartTokenizerFast, BertTokenizer, TFBertModel, RobertaTokenizer, TFRobertaModel
import sys
import json
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
gpu = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
print('TF version:', tf.__version__)
simplefilter('ignore')

string = ' '.join(sys.argv[2:])

key = sys.argv[1]

def test(string):

    #########################################################################################################################

    ## choice of tokenizer and base transformer model # do not change

    #########################################################################################################################

    #tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased-whole-word-masking")
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    #model = TFBartModel.from_pretrained('facebook/bart-large')
    model = TFBertModel.from_pretrained("bert-large-cased-whole-word-masking")
    #model = TFRobertaModel.from_pretrained('roberta-large')


    #########################################################################################################################

    ## gpu check # number of GPUs available should be atleast one

    #########################################################################################################################

    gpu = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))

    #if len(gpu) > 0:
    #    tf.config.experimental.set_memory_growth(gpu[0], True)

    #########################################################################################################################

    ## token embedder # do not change

    #########################################################################################################################

    def model_embeddings(sentence):
        inputs = tokenizer(sentence, return_tensors="tf")
        outputs = model(**inputs)
        outs = [item for item in outputs.values()]
    
        new_token_list, new_tensor_list = [], []
        for token, tensor in zip(tokenizer.tokenize(sentence), tf.squeeze(outputs.last_hidden_state)):
            if not '##' in token:
                new_token_list.append(token)
                new_tensor_list.append(tensor)
            else:
                new_token_list[-1] += token[2:]
                new_tensor_list[-1] += tensor

    
        return new_token_list, new_tensor_list


    #string = 'The refund process was pretty smooth. Customer service was great. The app was pretty good though. I would recommend it to anyone. The prices were fair.'
    token_list, tensor_list = model_embeddings(string)

    X = tf.convert_to_tensor([tensor.numpy() for tensor in tensor_list])

    #########################################################################################################################

    ## output embedding decoder loading # do not change

    #########################################################################################################################

    encoded_classes = np.load('encoded_classes_'+ key + '.npy')
    encoded_sentiments = np.load('encoded_sentiments_'+ key + '.npy')

    #########################################################################################################################

    ## decoder-only model loading (to act on token embeddings)

    #########################################################################################################################


    inputs = Input(shape=(X.shape[1],))
    hidden_1 = GaussianNoise(0.001)(inputs)
    hidden_1 = Dense(100, activation="relu")(inputs)
    hidden_1 = GaussianNoise(0.001)(hidden_1)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.5)(hidden_1)
    hidden_1 = Dense(10, activation="relu")(hidden_1)
    hidden_1 = GaussianNoise(0.001)(hidden_1)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.3)(hidden_1)
    outputs = Dense(encoded_classes.shape[0] +  encoded_sentiments.shape[0], activation="softmax")(hidden_1)
    model = Model(inputs=inputs, outputs=outputs, name="pt_bert_model")

    model.load_weights('./checkpoints_'+ key)

    #########################################################################################################################

    ## output_array # the final output of this code

    #########################################################################################################################

    preds = model.predict(X)
    ref = preds[:, :-encoded_sentiments.shape[0]]
    b = np.zeros_like(ref)
    b[np.arange(len(ref)), ref.argmax(1)] = 1
    ref = preds[:, -encoded_sentiments.shape[0]:]
    c = np.zeros_like(ref)
    c[np.arange(len(ref)), ref.argmax(1)] = 1
    preds_abs = np.hstack((b, c))


    preds_abs = np.vstack((
        encoded_classes[np.argmax(preds_abs[:, :-encoded_sentiments.shape[0]], axis = 1)], 
        encoded_sentiments[np.argmax(preds_abs[:, -encoded_sentiments.shape[0]:], axis = 1)], 
        np.max(preds[:, :-encoded_sentiments.shape[0]], axis = 1), 
        np.max(preds[:, -encoded_sentiments.shape[0]:], axis = 1))).T
    
    preds_abs = np.array([elem for elem in preds_abs if ('not' in elem[0] and elem[1] == 'na') or (not 'not' in elem[0] and elem[1] != 'na')])

    stop_words = set(stopwords.words('english'))
    output = []
    for token, pred, pos_tup in zip(token_list, preds_abs, nltk.pos_tag(token_list)):
        if float(pred[2]) >= 0.0:
            if not token.lower() in stop_words:
                if 'NN' in pos_tup[1] or 'VB' in pos_tup[1]:
                    output_ = {
                            'token': token.lower(), 
                            'class': pred[0], 
                            'sentiment': pred[1], 
                            'prob_class': float(pred[2]), 
                            'prob_sentiment': float(pred[3]),
                            'noun_type': pos_tup[1]
                            }
                    output.append(output_)

    output.append(TextBlob(string).sentiment.polarity)
    with open('output.json', 'w') as file:
        file.write(json.dumps(output, indent=4))    
    #print(output)

    #########################################################################################################################

    ## optional

    #########################################################################################################################
    #for token, category in output_array:
    #    print(token + '\t\t' + category)
    #print(X.shape)
    return output
test(string)

