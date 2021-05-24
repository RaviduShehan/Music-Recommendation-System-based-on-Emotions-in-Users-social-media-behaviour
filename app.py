
#Import the relevent Libraries
import time
import urllib.request
import tweepy
import pandas as pd
import numpy as np
import re
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,LSTM,GRU,Dense
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, flash
from input import UserInput


#imports for lyrics model
import nltk
import ast
from nltk.corpus import stopwords
from collections import Counter
from random import randint

#Twitter API credentials
consumer_key = 'Qp4hH6xgMKjla17LGjMJf4z6C'
consumer_code = 'a9bwmpwfAkLLCiP8Slmtq2JlpFAax1NJcmF1sVJ4ZM2rRgfHk2'
access_token = '2905175149-PbaI3EY0ZMdpibDIXgVS9yOA7NUjcrJPeLJx60z'
access_code = 'LW9tGNKTzt4KbReQ3LInTBMNHwWY6uDN0hMFJBtzQhvha'

#Authentication of the Tweeter credentials
authenticate = tweepy.OAuthHandler(consumer_key,consumer_code)

#set the access token
authenticate.set_access_token(access_token,access_code)

#Create the API object
api = tweepy.API(authenticate, wait_on_rate_limit=True)


fileD= open('TextModelData/train.txt','r')
x_train = []
y_train = []
for i in fileD:
    l = i.split(';')
    y_train.append(l[1].strip())
    x_train.append(l[0])
fileD= open('TextModelData/test.txt','r')
x_test =[]
y_test =[]
for i in fileD:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
filedD =open('TextModelData/val.txt','r')
for i in filedD:
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

#function to import the 1million word vectors ass for embedding purpose
def embedding_matrix(filepath,word_index,embedded_num_dims):
    vocab_size=len(word_index)+1
    embedding_matrix=np.zeros((vocab_size,embedded_num_dims))
    with open(filepath,encoding="utf8") as f:
        for line in f:
            word,*vector=line.split()
            if word in word_index:
                idx=word_index[word]
                embedding_matrix[idx] = np.array(vector,dtype=np.float32)[:embedded_num_dims]
    return embedding_matrix
#this word vectors used to train the model with more efficent
fname='embeddings/wiki-news-300d-1M.vec'
embedd_matrix=embedding_matrix(fname,index_of_words,embedded_num_dims)

#Create the embedding layer
embedd_layer=Embedding(vocab_size,embedded_num_dims,input_length=max_sequence_len,weights=[embedd_matrix],trainable=False)
gru_output_size=128
bidirectional=True
model=Sequential()
model.add(embedd_layer)
model.add(Bidirectional(GRU(units=gru_output_size,dropout=0.2,recurrent_dropout=0.2)))
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



#########Lyrics model

# Creat a tuple to contain the lyrics and emotion
def tuples(fileName):
    fileOpen = open(fileName)
    readFile = fileOpen.read()
    collection = ast.literal_eval(readFile)
    result = []
    for i in collection:
        details = (i["lyric"], i["sentiment"])
        result.append(details)
    return result


# Clean the training lyrics
def stop_words_removing(lyrics):
    word_array = []
    for i in lyrics:
        sentiment = i[1]
        stop_word = set(stopwords.words('english'))
        filtered_words = [word for word in i[0].split() if word not in stop_word]
        join_words = (filtered_words, sentiment)
        word_array.append(join_words)
    return word_array


# get the lyrics
def get_lyrics(lyrics):
    words_in_lyrics = []  # set the all words in to array
    for (words, sentiment) in lyrics:
        words_in_lyrics.extend(words)
    return words_in_lyrics


# function to check the word frequency and distribution
def word_features(word_list):
    word_list = nltk.FreqDist(word_list)
    word_features = word_list.keys()
    return word_features


# Extract Features from the words
def extract_features(document):
    document_words = set(document)
    features_in_doc = {}
    for word in word_features:
        features_in_doc['contains(%s' % word] = (word in document_words)
    return features_in_doc


# Removing Stop words from user recent songs
def stop_word_from_user(lyrics):
    stop_words_user = set(stopwords.words('english'))
    filtered_words = [word for word in lyrics.split() if word not in stop_words_user]
    return filtered_words


# Get the users emotions lyrics as a list
def user_emotions(file):
    file_open = open(file)
    file_read = file_open.read()
    file_split = file_read.split("|")
    result_list = []
    for lyrics in file_split:
        filtered_lyrics = stop_word_from_user(lyrics)
        output = classifier.classify(extract_features(filtered_lyrics))
        result_list.append(output)
    return result_list


# Functions to get the recommendations according to emotion
def recommendSongs(songsRecords):
    counter = Counter(songsRecords)
    train_file_open = open('LyricsDataset/training_original.txt')
    train_file_read = train_file_open.read()
    collection = ast.literal_eval(train_file_read)
    if counter['P'] >= counter['N']:
        positive_songs = [i for i in collection if i["sentiment"] == 'P']
        return positive_songs[randint(0, len(positive_songs) - 1)]["name"]
    else:
        negative_songs = [i for i in collection if i["sentiment"] == 'N']
    return negative_songs[randint(0, len(negative_songs) - 1)]["name"]


# Train the model
# create tuple and set path to train set
lyrics = tuples('LyricsDataset/training_original.txt')
filtered_corpus = stop_words_removing(lyrics)

word_features = word_features(get_lyrics(filtered_corpus))
# applying the features
training_set = nltk.classify.apply_features(extract_features, filtered_corpus)
# adding to classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the model
test_lyrics = tuples('LyricsDataset/testing_original.txt')
test_corpus = stop_words_removing(test_lyrics)
test_set = nltk.classify.apply_features(extract_features, test_corpus)



def sentiment(output):
    #Assigning to Emotional categories
    if str(output) == "joy":
        return "Neutral"
    if str(output) == "anger":
        return "Positive"
    if str(output) == "sadness":
        return "Positive"
    if str(output) == "fear":
        return "Positive"
    if str(output) == "surprise":
        return "Neutral"
    if str(output) == "love":
        return "Positive"

#Defined variables to trigger the emotion count
joyVal = 0
sadVal = 0
loveVal = 0
fearVal = 0
surpriseVal = 0
angerVal = 0
emoCountList=[]

def englishPrediction(tweetsStr):

    global sadVal
    global joyVal
    global loveVal
    global fearVal
    global surpriseVal
    global angerVal

    from tensorflow import keras
    loaded_model = keras.models.load_model("embeddings/savedModelD")
    message = [tweetsStr]
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq)
    pred = loaded_model.predict(padded)
    emotion = class_names[np.argmax(pred)]
    print('Message:' + str(message))

    #Update the count of emotion in each tweets in the day
    if emotion == 'joy':
        joyVal = joyVal + 1
    elif emotion == 'sadness':
        sadVal = sadVal + 1
    elif emotion == 'love':
        loveVal += 1
    elif emotion == 'surprise':
        surpriseVal += 1
    elif emotion == 'anger':
        angerVal += 1
    elif emotion == 'fear':
        fearVal += 1
    emoCountList = [joyVal, sadVal, loveVal, surpriseVal, angerVal, fearVal]
    maximumVal = max(emoCountList)
    indexofMax = emoCountList.index(maximumVal)

    if indexofMax == 0:
        emotionA = "Joy"
    elif indexofMax == 1:
        emotionA = "Sadness"
        print("Test", emotionA)
    elif indexofMax == 2:
        emotionA = "Love"
    elif indexofMax == 3:
        emotionA = "Surprise"
    elif indexofMax == 4:
        emotionA = "Anger"
    elif indexofMax == 5:
        emotionA = "Fear"
    print(emoCountList)
    file_open = open('LyricsDataset/userSongs.txt')
    file_read = file_open.read()
    file_split = file_read.split("|")
    result_list = []

    emotion_details = user_emotions('LyricsDataset/userSongs.txt')
    recommendation = recommendSongs(emotion_details)
    print("Recommend song according to mood : " + recommendation)
    print('Emotion:', emotionA)
    return emotionA,recommendation

def sinhalaPredictions(tweetsStr):

    global sadVal
    global joyVal
    global loveVal
    global fearVal
    global surpriseVal
    global angerVal
    from tensorflow import keras
    loaded_model = keras.models.load_model("embeddings/SinhalaText22A")
    message = [tweetsStr]
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_sequence_len)
    pred = model.predict(padded)
    emotion = class_names[np.argmax(pred)]

    # Update the count of emotion in each tweets in the day
    if emotion == 'joy':
        joyVal = joyVal + 1
    elif emotion == 'sadness':
        sadVal = sadVal + 1
    elif emotion == 'love':
        loveVal += 1
    elif emotion == 'surprise':
        surpriseVal += 1
    elif emotion == 'anger':
        angerVal += 1
    elif emotion == 'fear':
        fearVal += 1
    emoCountList = [joyVal, sadVal, loveVal, surpriseVal, angerVal, fearVal]
    maximumVal = max(emoCountList)
    indexofMax = emoCountList.index(maximumVal)

    if indexofMax == 0:
        emotionA = "Joy"
    elif indexofMax == 1:
        emotionA = "Sadness"
        print("Test", emotionA)
    elif indexofMax == 2:
        emotionA = "Love"
    elif indexofMax == 3:
        emotionA = "Surprise"
    elif indexofMax == 4:
        emotionA = "Anger"
    elif indexofMax == 5:
        emotionA = "Fear"
    print(emoCountList)

    file_open = open('LyricsDataset/userSongs.txt')
    file_read = file_open.read()
    file_split = file_read.split("|")
    result_list = []

    emotion_details = user_emotions('LyricsDataset/userSongs.txt')
    recommendation = recommendSongs(emotion_details)
    print(message)

    #Return the emotion and recommendation to submit function
    return emotionA,recommendation


app = Flask(__name__)
app.config['SECRET_KEY'] ='ravidu'

@app.route('/', methods=['GET' ,'POST'])
def submit():
    form =UserInput()
    if form.is_submitted():
        result= request.form['username']
        userIDA= str(result)
        if(userIDA ==""):
            flash("Please enter username")
            return render_template('index.html', form=form)
        else:
            userId = userIDA
            print("useride", userId)
            print("Recent tweets from: ", userId)
            try:
                recentPosts = api.user_timeline(screen_name=userId, lang="sn", tweet_mode="extended")
            except tweepy.TweepError as e:
                flash("Invalid username")
                return render_template('index.html', form=form)
            isCompleted = 0 #check the previous conditions are met
            ifAvailable =0
            emo=""
            reco=""
            for tweet in recentPosts:
                language = tweet.lang
                if (datetime.datetime.now() - tweet.created_at).days < 1:
                    ifAvailable=1
                    #Tweets cleaning
                    tweetAsStr = tweet.full_text
                    tweetAsStr = re.sub(r'@[A-Za-z0-9]+', '', tweetAsStr)  # remove the mentions in the tweets
                    tweetAsStr = re.sub(r'#', '', tweetAsStr)  # remove the hashtags
                    tweetAsStr = re.sub(r'RT[\s]+', '', tweetAsStr)  # Remove the RT
                    tweetAsStr = re.sub(r'https?:\/\/\S+', '', tweetAsStr)  # Remove the urls

                    #get the english tweets
                    if (language == "en"):
                        englishTweets = tweetAsStr
                        emo,reco=englishPrediction(englishTweets)
                        songUrl = getYoutubeLink(reco)
                        isCompleted=1
                    #check the language for sinhala
                    elif(language == "si"):
                        sinhalaTweets = tweetAsStr
                        emo, reco = sinhalaPredictions(sinhalaTweets)
                        songUrl = getYoutubeLink(reco)
                        isCompleted =1
                    elif(isCompleted != 1):
                        print("Twitter lang: ",language)
                        #If other language expect english and sinhala detected
                        flash("Your tweet languge is does not support to the system")
                        return render_template('index.html', form=form)
                elif(ifAvailable !=1):
                    #If the tweets are not posted in past 24 hours
                    flash("No Recent Tweets available, Please update your profile and try again")
                    return render_template('index.html',form=form)

            return render_template('dashboard.html', userId=userId, emotion=emo, recommendation=reco,
                           songLink=songUrl)
    return render_template('index.html', form=form)

def getYoutubeLink(songName):
    #Get the recommended song name
    recommendedName = songName
    suggestedName = recommendedName.replace(" ", "+")

    #Create a search request to get the relevent songs from youtube
    htmlforS = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + suggestedName)

    #Define the video link
    videoId = re.findall(r"watch\?v=(\S{11})", htmlforS.read().decode())
    songUrl="https://www.youtube.com/watch?v=" + videoId[0]

    #SongUrl consists with the youtube links
    return songUrl


if __name__ == '__main__':
    app.run(debug=True,threaded=True)
