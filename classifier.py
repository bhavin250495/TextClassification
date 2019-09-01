import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import re
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,SpatialDropout1D,LSTM,Dense,Conv1D,MaxPooling1D,Flatten
from keras.callbacks import EarlyStopping
from keras.metrics import binary_crossentropy,categorical_crossentropy,categorical_accuracy
from sklearn.metrics import log_loss
from sklearn.utils import shuffle

## PATHS
train_combined = 'participants/train/extracted_data/extract_combined.csv'
train_label = 'participants/train/labels/labels.csv'
sample_submission = 'participants/sample_submission.csv'

## LOAD DATAFRAME
sample_submission_df = pd.read_csv(sample_submission)
train_data = pd.read_csv(train_combined)
train_label = pd.read_csv(train_label,usecols=['document_name','is_fitara'])


### Final df
training_data = pd.merge(train_data,train_label,on='document_name',
    how='inner')


## Create testing data unknown
testing_data = training_data.iloc[:10,:]
training_data = training_data.drop([10])

## Avoid overfitting by duplicating data
def expand_data(training_data):

    is_fitara,non_fitara = training_data.groupby(['is_fitara'])

    ### Clssified data
    non_fitara_df = is_fitara[1]
    #(681, 3)

    is_fitara_df = non_fitara[1]
    #(274, 3)

    ### Avoid overfitting by duplicating data
    is_fitara_df =  is_fitara_df.append([is_fitara_df]*1,ignore_index=True)
    training_data = pd.concat([is_fitara_df,non_fitara_df])
    training_data = training_data.append([training_data]*1,ignore_index=True)

    training_data = shuffle(training_data)
    return training_data


training_data = expand_data(training_data)

####################################################### Word embeddings

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


training_data['text'] = training_data['text'].apply(clean_text)
training_data['text'] = training_data['text'].str.replace('\d+', '')


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(training_data['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(training_data['text'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

Y = pd.get_dummies(training_data['is_fitara']).values

####################################################### Word embeddings


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4, random_state = 42)

## Deep ML models

def getLSTMModel(input_shape):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_shape))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
    return model


def getCNNModel(input_shape):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_shape))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def validate_model(model,x_test,y_test):
    y_pred = model.predict(x_test)
    accr = model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    loss = log_loss(y_test, y_pred, eps=1e-15)
    print('Log loss  :: {}'.format(loss))

epochs = 20
batch_size = 64
model = getCNNModel(X_train.shape[1])
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

validate_model(model,x_test=X_test,y_test=Y_test)

def save_submission(model,file_name = 'submission'):
    test = 'participants/test/extracted_data/extract_combined.csv'
    test_df = pd.read_csv(test)
    test_x = tokenizer.texts_to_sequences(test_df['text'].values)
    test_x = pad_sequences(test_x, maxlen=MAX_SEQUENCE_LENGTH)
    pred_x = model.predict(test_x)
    test_df['pred_fitara'] = pred_x[:, 0]
    test_df = test_df.drop(['text'], axis=1)
    test_df.to_csv(file_name + '.csv')

save_submission(model,'v1')







import matplotlib.pyplot as plt
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();

#
# for i in range(10):
#     new_complaint = [testing_data['text'][i]]
#     seq = tokenizer.texts_to_sequences(new_complaint)
#     padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
#     pred = model.predict(padded)
#
#
#     labels = [ 'non_fitara','is_fitara']
#     import numpy as np
#     print(i)
#
#     print(pred, labels[np.argmax(pred)])
