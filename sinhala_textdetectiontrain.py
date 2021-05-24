
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,LSTM,GRU,Dense
from nltk.tokenize import word_tokenize



#Extract the data set from txt files and put them in data frames

sinfileData= open('TextModelData/nSinhala/train.txt','r')
x_train = []
y_train = []
for i in sinfileData:
    l = i.split(';')
    y_train.append(l[1].strip())
    x_train.append(l[0])
sinfileData= open('TextModelData/nSinhala/test.txt','r')
x_test =[]
y_test =[]
for i in sinfileData:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
sinfileData =open('TextModelData/nSinhala/val.txt','r')
for i in sinfileData:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
sintrain_data=pd.DataFrame({'Text' : x_train, 'Emotion':y_train})
sintest_data=pd.DataFrame({'Text' : x_test, 'Emotion':y_test})
sindata=sintrain_data.append(sintest_data,ignore_index=True)


#clean the data set sentences for all test ,train and validation data

# clean the data set sentences for all test ,train and validation data
def texts_cleaning(sindata):
    sindata = re.sub(r"(#[\d\w\.]+)", '', sindata)
    sindata = re.sub(r"(@[\d\w\.]+)", '', sindata)
    sindata = word_tokenize(sindata)

    return sindata

sintexts_clened = [' '.join(texts_cleaning(text)) for text in sindata.Text]
sintraining_texts = [' '.join(texts_cleaning(text)) for text in x_train]
sintesting_texts = [' '.join(texts_cleaning(text)) for text in x_test]

#Tokenize the words to extract unique word and added to dictionary
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sintexts_clened)
sintrain_sequence=tokenizer.texts_to_sequences(sintraining_texts)
sintest_sequence=tokenizer.texts_to_sequences(sintesting_texts)
index_of_words=tokenizer.word_index
vocab_size=len(index_of_words)+1


#num classes is 6 since we identify the 6 unique emotional words
number_of_classes=6
embedded_num_dims=300
max_sequence_len=500
class_names=['anger','sadness','fear','joy','surprise','love']
X_train_pad=pad_sequences(sintrain_sequence,maxlen=max_sequence_len)
X_test_pad=pad_sequences(sintest_sequence,maxlen=max_sequence_len)

print(X_train_pad)


#emotional words added to 0-5 categorical values to encode the dictionary
encoding={'anger':0,'sadness':1,'fear':2,'joy':3,'surprise':4,'love':5}
y_train=[encoding[x] for x in sintrain_data.Emotion]
y_test=[encoding[x] for x in sintest_data.Emotion]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#function to import the 1million word vectors ass for embedding purpose
def create_embedding_matrix_forSinhala(filepath,word_index,embedded_num_dims):
    vocabulary_size=len(word_index)+1
    embedding_matrix=np.zeros((vocabulary_size,embedded_num_dims))
    with open(filepath,encoding="utf8") as f:
        for line in f:
            word,*vector=line.split()
            if word in word_index:
                idOfWord=word_index[word]
                embedding_matrix[idOfWord] = np.array(vector,dtype=np.float32)[:embedded_num_dims]
    return embedding_matrix

#Assign the word vecotor
wordVector='embeddings/wiki-news-300d-1M.vec'
embedd_matrix=create_embedding_matrix_forSinhala(wordVector,index_of_words,embedded_num_dims)

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
checkpoint = ModelCheckpoint('embeddings/SinhalaText22A')
batch_size=128
epochs=30
hist=model.fit(X_train_pad,y_train,batch_size=batch_size,epochs=epochs,callbacks=[checkpoint],validation_data=(X_test_pad,y_test))

model.save('embeddings/SinhalaText1')
