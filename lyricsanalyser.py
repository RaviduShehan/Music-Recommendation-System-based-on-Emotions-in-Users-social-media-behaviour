
import nltk
nltk.download('stopwords')
import ast
from nltk.corpus import stopwords
from collections import Counter
from random import randint


def tuples(fileName):
  fileOpen = open(fileName)
  readFile = fileOpen.read()
  collection = ast.literal_eval(readFile)
  result = []
  for i in collection:
    details = (i["lyric"], i["sentiment"])
    result.append(details)
  return result

#Clean the training lyrics
def stop_words_removing(lyrics):
  word_array = []
  for i in lyrics:
    sentiment = i[1]
    stop_word = set(stopwords.words('english'))
    filtered_words = [word for word in i[0].split() if word not in stop_word]
    join_words = (filtered_words, sentiment)
    word_array.append(join_words)
  return word_array

#get the lyrics
def get_lyrics(lyrics):
  words_in_lyrics = [] # set the all words in to array
  for (words, sentiment) in lyrics:
    words_in_lyrics.extend(words)
  return words_in_lyrics

#function to check the word frequency and distribution
def word_features(word_list):
  word_list = nltk.FreqDist(word_list)
  word_features = word_list.keys()
  return word_features

#Features extraction
def extract_features(document):
  document_words = set(document)
  features_in_doc ={}
  for word in word_features:
    features_in_doc['contains(%s' %word] = (word in document_words)
  return features_in_doc


#Get the users emotions lyrics as a list
def user_emotions(file):
    file_open = open(file)
    file_read = file_open.read()
    file_split=file_read.split("|")
    result_list=[]
    for lyrics in file_split:
        filtered_lyrics=stop_words_removing(lyrics)
        output = classifier.classify(extract_features(filtered_lyrics))
        result_list.append(output)
    return result_list

#Functions to get the recommendations according to emotion
def recommendSongs(songsRecords):
  counter = Counter(songsRecords)
  train_file_open = open('LyricsDataset/training_original.txt')
  train_file_read = train_file_open.read()
  collection =ast.literal_eval(train_file_read)
  if counter['P'] >= counter['N']:
    positive_songs = [i for i in collection if i["sentiment"] == 'P']
    return positive_songs[randint(0,len(positive_songs)-1)]["name"]
  else:
   negative_songs = [i for i in collection if i["sentiment"] == 'N']
  return negative_songs[randint(0,len(negative_songs)-1)]["name"]

#Train the model
#create tuple and set path to train set
lyrics = tuples('LyricsDataset/training_original.txt')
filtered_corpus = stop_words_removing(lyrics)

word_features = word_features(get_lyrics(filtered_corpus))
#applying the features
training_set = nltk.classify.apply_features(extract_features,filtered_corpus)
#adding to classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

#Test the model
test_lyrics = tuples('LyricsDataset/testing_original.txt')
test_corpus = stop_words_removing(test_lyrics)
test_set =nltk.classify.apply_features(extract_features,test_corpus)

print("Accuracy :" + str(nltk.classify.accuracy(classifier, test_set)))


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



#Recommendation
 
file_open = open('LyricsDataset/userSongs.txt')
file_read = file_open.read()
file_split=file_read.split("|")
result_list=[]

emotion_details= user_emotions('LyricsDataset/userSongs.txt')
print("Recommend song according to mood : " + recommendSongs(emotion_details) )