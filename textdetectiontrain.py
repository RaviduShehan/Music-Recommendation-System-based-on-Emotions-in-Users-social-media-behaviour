
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,LSTM,GRU,Dense
import nltk
from nltk.tokenize import word_tokenize
import warnings
nltk.download('punkt')
warnings.filterwarnings('ignore')

#Extract the data set from txt files and put them in data frames
fileData= open('TextModelData/train.txt','r')
x_train = []
y_train = []
for i in fileData:
    l = i.split(';')
    y_train.append(l[1].strip())
    x_train.append(l[0])
fileData= open('TextModelData/test.txt','r')
x_test =[]
y_test =[]
for i in fileData:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
fileData =open('TextModelData/val.txt','r')
for i in fileData:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
train_data=pd.DataFrame({'Text' : x_train, 'Emotion':y_train})
test_data=pd.DataFrame({'Text' : x_test, 'Emotion':y_test})
data=train_data.append(test_data,ignore_index=True)

#clean the data set sentences for all test ,train and validation data
def texts_cleaning(data):
    data=re.sub(r"(#[\d\w\.]+)", '', data)
    data=re.sub(r"(@[\d\w\.]+)", '', data)
    data=word_tokenize(data)
    print(data)
    return data
texts_clened=[' '.join(texts_cleaning(text)) for text in data.Text]
training_texts=[' '.join(texts_cleaning(text)) for text in x_train]
testing_texts=[' '.join(texts_cleaning(text)) for text in x_test]

#Tokenize the words to extract unique word and added to dictionary
tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts_clened)
train_sequence=tokenizer.texts_to_sequences(training_texts)
test_sequence=tokenizer.texts_to_sequences(testing_texts)
index_of_words=tokenizer.word_index
vocab_size=len(index_of_words)+1



#num classes is 6 since we identify the 6 unique emotional words
number_of_classes=6
embedded_num_dims=300
max_sequence_len=500
class_names=['anger','sadness','fear','joy','surprise','love']
X_train_pad=pad_sequences(train_sequence,maxlen=max_sequence_len)
X_test_pad=pad_sequences(test_sequence,maxlen=max_sequence_len)



#emotional words added to 0-5 categorical values to encode the dictionary
encoding={'anger':0,'sadness':1,'fear':2,'joy':3,'surprise':4,'love':5}
y_train=[encoding[x] for x in train_data.Emotion]
y_test=[encoding[x] for x in test_data.Emotion]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#function to import the 1million word vectors as for embedding purpose
def create_embedding_matrix_for_English(filepath,word_index,embedded_num_dims):
    vocabulary_size=len(word_index)+1
    embedding_matrix=np.zeros((vocabulary_size,embedded_num_dims))
    with open(filepath,encoding="utf8") as f:
        for line in f:
            word,*vector=line.split()
            if word in word_index:
                idOfWord=word_index[word]
                embedding_matrix[idOfWord] = np.array(vector,dtype=np.float32)[:embedded_num_dims]
    return embedding_matrix
###### Word vector extracted from T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations###########
wordVector='embeddings/wiki-news-300d-1M.vec'
embedd_matrix=create_embedding_matrix_for_English(wordVector,index_of_words,embedded_num_dims)

#Create the embedding layer
embedd_layer=Embedding(vocab_size,embedded_num_dims,input_length=max_sequence_len,weights=[embedd_matrix],trainable=False)
gru_output_size=128
bidirectional=True
model=Sequential()
model.add(embedd_layer)
model.add(Bidirectional(GRU(units=gru_output_size,dropout=0.2,recurrent_dropout=0.2)))
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



#Train model for 30 epcohs
from tensorflow.python.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('embeddings/savedModelC')
batch_size=128
epochs=30
hist=model.fit(X_train_pad,y_train,batch_size=batch_size,epochs=epochs,callbacks=[checkpoint],validation_data=(X_test_pad,y_test))

model.save('embeddings/savedModelC')
