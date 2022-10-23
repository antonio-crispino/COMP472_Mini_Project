import os
import gzip
import json
import numpy as np
from gensim.downloader import load
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

# declare os paths
main_directory = os.path.join(os.getcwd(), 'Mini_Project_1')
dataset_path = os.path.join(main_directory, 'Dataset', 'goemotions.json.gz')

# load posts into an array
posts_array = np.array([])
with gzip.open(dataset_path, 'rb') as f:
    posts_array = np.array(json.load(f))

# arrays for contents, emotions, and sentiments
content_array = np.array([post[0] for post in posts_array])
emotion_array = np.array([post[1] for post in posts_array])
sentiment_array = np.array([post[2] for post in posts_array])

# ----------
# Part 3.1
# ----------

pretrained_embedding_model = load('word2vec-google-news-300')
# print(pretrained_embedding_model) #TESTING

# ----------
# Part 3.2
# ----------

# if 'punkt' is not downloaded, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# create 2d-array where sub-arrays are tokens from each post (content)
tokenized_content_array = [word_tokenize(content) for content in content_array]
# flatten 2d-array to create an array of every token from every sub-array
all_word_tokens = np.array([token for tokenized_content in tokenized_content_array for token in tokenized_content])
# Number of tokens in training set
print("Number of tokens in training set: ", len(all_word_tokens))

# print(tokenized_content_array) #TESTING
# print(all_word_tokens) #TESTING


# ----------
# Part 3.3
# ----------

# Take sampel reddit post from the data set
redditpostsample = ['That', 'looks', 'amazing']

# Compute the embeddings individually per word

# Two arrays, one for vectorized
vecter_list = []  # will return embeddings for each word as an array one after the other
word_filtered_list = []  # check if words exist in other posts (otherwise skip), if so, returns same value as redditpostsample

vecter_list = [pretrained_embedding_model[word] for word in redditpostsample if
               word in pretrained_embedding_model.index_to_key]
word_filtered_list = [word for word in redditpostsample if word in pretrained_embedding_model.index_to_key]

# print(vecter_list) #TESTING
# print(word_filtered_list) #TESTING

# Create a data frame (puts data in clean table) using pandas
data_frame = pd.DataFrame.from_dict(dict(zip(word_filtered_list, vecter_list)), orient='index')

# Write the new dataframe file to a .json
json.dump(data_frame.to_dict(), open("embeddings_of_post.json", 'w'))
json_embedded_posts = pd.read_json("embeddings_of_post.json")

# Turn tokens using JSON dictionary format
embedded_posts_tokens = json_embedded_posts.to_dict()
json_embedded_posts.head(3)

# Display the data in console
# display(data_frame) #TESTING

# Compute the embeddings as an average
average_embeddings = []
for i in range(0, pretrained_embedding_model.vector_size - 1):
    k = 0
    for j in range(0, len(embedded_posts_tokens[0]) - 1):
        k += json_embedded_posts[i][j]
    average_embeddings.append(k / len(embedded_posts_tokens[0]))

# Create a data frame (puts data in clean table) using pandas
data_frame = pd.DataFrame(average_embeddings, columns=['average'])
data_frame.T.head()

display(data_frame)  # TESTING


# ----------
# Part 3.4
# ----------

# Retrain model
# Taken from Part 2.2
def part_2_2(x):
    '''
    Function splits dataset for train and test (80% - 20%)
    :param: Independant variable (content)
    :return: x_train, x_test, ye_train, ye_test, ys_train, ys_test
  '''
    ye = emotion_array  # dependent variable EMOTION
    ys = sentiment_array  # dependent variable SENTIMENT
    x_train, x_test, ye_train, ye_test, ys_train, ys_test = train_test_split(x, ye, ys, test_size=0.2, random_state=2)
    return x_train, x_test, ye_train, ye_test, ys_train, ys_test


# the train and test sets of the
x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2(content_array)


def hit_rate(emb_model, content):
    '''
      Function that calculates the hit rate for the split dataset for train and test (80% - 20%)
      :param: emb_model (Word2vec model), content (data_set as 2D array of phrases format),
      :return: x_train, x_test, ye_train, ye_test, ys_train, ys_test
    '''

    # create two sets that will collect the words sorted
    vocabulary_found = set()
    other_vocabulary = set()

    # Loop through each phrase
    for phrase in content:
        # Split phrase arrays and loop
        for word in phrase.split():
            word = word.lower()
            vocabulary_found.add(word)
            # check if the word emmbedding is found or not
            try:
                temp = emb_model[word]
            # If not found, add to other array
            except:
                if (word not in other_vocabulary):
                    other_vocabulary.add(word)
    # compute hit rate as a percentage
    return (float(len(vocabulary_found) - len(other_vocabulary))) * 100.0 / float(len(vocabulary_found))


# Print the Hit Rates
print("Training Set Hit Rate: {0:.2f}%".format(hit_rate(pretrained_embedding_model, x_train)))
print("Test Set Hit Rate: {0:.2f}%".format(hit_rate(pretrained_embedding_model, x_test)))


# ----------
# Part 3.5
# ----------
# part_2_3_3(f)


# Train MLP Base

# Using Part 2.3.3
def part_2_3_3(f):
    '''
    Function to for MLP classification
    outputs BASE-MLP data: EmotionScore, SentimentScore, Classification Report
  '''

    # Max iteration chosen to be small to reduce runtime
    emotion_classifier = MLPClassifier(activation='logistic', max_iter=2)
    emotion_model = emotion_classifier.fit(x_train, ye_train)
    emotion_prediction = emotion_model.predict(x_test)
    emotion_accuracy = accuracy_score(ye_test, emotion_prediction)

    # Max iteration chosen to be small to reduce runtime
    sentiment_classifier = MLPClassifier(activation='logistic', max_iter=2)
    sentiment_model = sentiment_classifier.fit(x_train, ys_train)
    sentiment_prediction = sentiment_model.predict(x_test)
    sentiment_accuracy = accuracy_score(ys_test, sentiment_prediction)

    # For part 2.4
    f.write(f"====================================== BASE-MLP ======================================\n\n")
    f.write(f"Emotion Score: {emotion_accuracy}\n\n")
    f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction, zero_division=1)}\n")
    f.write(f"Sentiment Score: {sentiment_accuracy}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")
