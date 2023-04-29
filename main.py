import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
string.punctuation
import os
import sys
import csv
import warnings
import re
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import nltk
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import SGD

nltk.download('stopwords')

filename = 'neg_data.csv'
header = ['tweet', 'status']
cwd = os.getcwd()


def readling_data():
    path = cwd+'\datasettrain\/test\/neg'
    dir_list = os.listdir(path)
    # print(dir_list)
    with open(filename, 'w', newline='') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(header) # 4. write the header
        for i in dir_list:
            file1 = open(path+'/'+i,"r",errors='ignore')
            lines = file1.readlines()
            csvwriter.writerow([lines[0], "negative"])
            file1.close

    path = cwd+'\datasettrain\/test\pos'
    dir_list = os.listdir(path)
    with open(filename, 'a', newline='') as file:
        csvwriter = csv.writer(file)
        for i in dir_list:
            file1 = open(path+'/'+i,"r",errors='ignore')
            lines = file1.readlines()
            csvwriter.writerow([lines[0], "positive"])
            file1.close 
    
    path = cwd+'\datasettrain\/train\/neg'
    dir_list = os.listdir(path)
    with open(filename, 'a', newline='') as file:
        csvwriter = csv.writer(file)
        for i in dir_list:
            file1 = open(path+'/'+i,"r",errors='ignore')
            lines = file1.readlines()
            csvwriter.writerow([lines[0], "negative"])
            file1.close 

    path = cwd+'\datasettrain\/train\pos'
    dir_list = os.listdir(path)
    with open(filename, 'a', newline='') as file:
        csvwriter = csv.writer(file)
        for i in dir_list:
            file1 = open(path+'/'+i,"r",errors='ignore')
            lines = file1.readlines()
            csvwriter.writerow([lines[0], "positive"])
            file1.close 


# readling_data()

warnings.filterwarnings("ignore")


pd.options.display.max_colwidth = 1000
pd.options.display.max_rows  = 100
pd.set_option("display.min_rows", 200)

train = pd.read_csv("full-corpus.csv",encoding='latin-1')
train_data = train[["tweet","status"]]

test = pd.read_csv("live_data.csv",encoding='latin-1')
test_data = train[["tweet","status"]]


# print(train_data.head(10))
print(train_data.shape)

print("Count Of Labels : {} ".format(train_data.status.value_counts().to_dict()))
print("Total number of Labels : {}".format(len(train_data.status.unique())))

stop_words = stopwords.words('english')
train_data["tweet"]  = train_data['tweet'].apply(lambda x:x.lower())
test_data["tweet"]  = test_data['tweet'].apply(lambda x:x.lower())


def clean_data(text):
    text = str(text).strip()
    text = text.replace("?","")
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"@\w+","",text)
    text = re.sub(r"#\w+","",text)
    text = re.sub(r"\d+","",text)
    text = re.sub(r"<.*?>","",text)
    text = text.split()
    text = " ".join([word for word in text if not word in stop_words])
    #text - str(text).strip()
    return text

train_data["tweet"] = train_data['tweet'].apply(lambda x : clean_data(x))
test_data["tweet"] = test_data['tweet'].apply(lambda x : clean_data(x))


train_data['status'].unique()
l = dict()
for idx,lbl in enumerate(train_data['status'].unique()):
    l[lbl] = idx 

test_data['status'].unique()
l = dict()
for idx,lbl in enumerate(test_data['status'].unique()):
    l[lbl] = idx
    
train_data['status'] = train_data['status'].map(l)
test_data['status'] = test_data['status'].map(l)


# max_len = np.max(train_data['tweet'].apply(lambda x : len(x)))
max_len = 400

train,testn = train_test_split(train_data,test_size = 0.2)
test = test_data

X_train = train[["tweet"]]
y_train = train[["status"]]
X_test = test[['tweet']]
y_test = test[["status"]]


X_train.shape,y_train.shape,X_test.shape,y_test.shape

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train["tweet"])
vocab_length = len(tokenizer.word_index) + 1

x_train = tokenizer.texts_to_sequences(X_train["tweet"])
x_test = tokenizer.texts_to_sequences(X_test["tweet"])

x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

print("Vocab length:", vocab_length)
print("Max sequence length:", max_len)

embedding_dim = 16
num_epochs = 10

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_length,embedding_dim,input_length = max_len),
    tf.keras.layers.LSTM(128,return_sequences = True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2,activation = 'softmax')
])
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.summary()

tf.keras.utils.plot_model(model)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

batch_size = 30
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()