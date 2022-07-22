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

print('TF version:', tf.__version__)
simplefilter('ignore')

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

string = ' '.join(sys.argv[1:])

#string = 'The refund process was pretty smooth. Customer service was great. The app was pretty good though. I would recommend it to anyone. The prices were fair.'
token_list, tensor_list = model_embeddings(string)

X = tf.convert_to_tensor([tensor.numpy() for tensor in tensor_list])

#########################################################################################################################

## output embedding decoder loading # do not change

#########################################################################################################################

decoded_array = np.load('encoded_classes.npy')

#########################################################################################################################

## decoder-only model loading (to act on token embeddings)

#########################################################################################################################

inputs = Input(shape=(X.shape[1],))
hidden_1 = GaussianNoise(0.01)(inputs)
hidden_1 = Dense(100, activation="relu")(inputs)
hidden_1 = BatchNormalization()(hidden_1)
hidden_1 = Dropout(0.1)(hidden_1)
outputs = Dense(decoded_array.shape[0], activation="softmax")(hidden_1)
model = Model(inputs=inputs, outputs=outputs, name="pt_bert_model")
model.load_weights('./checkpoints')

#########################################################################################################################

## output_array # the final output of this code

#########################################################################################################################

output_array = [
        (
            token_list[enum], decoded_array[arg]
            ) for enum, arg in zip(
                range(model.predict(X).shape[0]), np.argmax(model.predict(X), axis = 1)
                )
            ]

#########################################################################################################################

## optional

#########################################################################################################################
for token, category in output_array:
    print(token + '\t\t' + category)
