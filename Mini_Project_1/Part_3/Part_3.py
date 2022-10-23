import os
import gzip
import json
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.downloader import load
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

# declare os paths
main_directory =  os.path.join(os.getcwd(), 'Mini_Project_1')
dataset_path = os.path.join(main_directory, 'Dataset', 'goemotions.json.gz')
output_path = os.path.join(main_directory, 'Part_2', 'Output.txt')

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

# --------------------
# Part 3.3 and 3.4
# --------------------

def part_3_3_and_3_4(tokenized_cont_arr):
  content_embeddings_obj = {} # an object containing objects with original_post, tokenized_post, post_embeddings, average_of_post_embeddings, hit_rate_of_post
  tokens_in_word2vec = 0 # the total number of tokens in all posts that had embeddings in Word2Vec
  total_tokens = 0 # the total number of tokens in all posts
  index = -1 # the current post index

  for tokenized_post in tokenized_cont_arr:
    index += 1
    post_embeddings_word2vec_obj = {}
    num_of_embeddings = 0 # total number of embeddings in post, and total number of tokens in post that are also in Word2Vec
    total_tokens += len(tokenized_post) # total tokens in post
    for token in tokenized_post:
      if token in pretrained_embedding_model: # proceed if the token exists in Word2Vec
        post_embeddings_word2vec_obj[token] = pretrained_embedding_model[token] # adding the embedding value to the token
        num_of_embeddings += 1
        tokens_in_word2vec += 1

    content_embeddings_obj[index] = {
      "tokenized_post": tokenized_post,
      "average_of_post_embeddings": np.mean(np.array([post_embeddings_word2vec_obj[token] for token in post_embeddings_word2vec_obj]), axis=0).tolist(),
      "hit_rate_of_post": 0 if len(tokenized_post) == 0 else num_of_embeddings/len(tokenized_post)
      }
    
  overall_hit_rate = 0 if total_tokens == 0 else tokens_in_word2vec/total_tokens
  content_embeddings_obj["overall_hit_rate"] = overall_hit_rate
  return content_embeddings_obj

# Part 2.2 for Part 3
def part_2_2(x):
  '''
    Function splits dataset for train and test (80% - 20%)
    :param: Independant variable (content)
    :return: x_train, x_test, ye_train, ye_test, ys_train, ys_test
  '''
  ye = emotion_array # dependent variable EMOTION
  ys = sentiment_array # dependent variable SENTIMENT
  x_train, x_test, ye_train, ye_test, ys_train, ys_test = train_test_split(x, ye, ys, test_size=0.2, random_state=2)
  return x_train, x_test, ye_train, ye_test, ys_train, ys_test

# the train and test sets of the 
x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2(tokenized_content_array)

# embeddings objects
content_embeddings_obj = part_3_3_and_3_4(tokenized_content_array)
x_train_embeddings_obj = part_3_3_and_3_4(x_train)
x_test_embeddings_obj = part_3_3_and_3_4(x_test)

print(content_embeddings_obj["overall_hit_rate"])
print(json.dumps(content_embeddings_obj[0], indent=2)) #example
print("The training set hit value: ")
print(x_train_embeddings_obj["overall_hit_rate"])
print(json.dumps(x_train_embeddings_obj[0], indent=2)) #example
print("The test set hit value: ")
print(x_test_embeddings_obj["overall_hit_rate"])
print(json.dumps(x_test_embeddings_obj[0], indent=2)) #example
# json.dump(content_embeddings_obj, open(os.path.join(main_directory, 'Part_3/embeddings', 'content_embeddings_obj.json'), 'w'), indent=2) #example


# Part 3.5 and 3.6
def part_3_5(f):
    '''
    Function to for MLP classification
    outputs BASE-MLP data: EmotionScore, SentimentScore, Classification Report
  '''

    # Max iteration chosen to be small to reduce runtime
    classifier_of_emotions_train = MLPClassifier(activation='logistic', max_iter=2)
    model_of_emotions_train = classifier_of_emotions_train.fit(x_train, ye_train)
    predictions_of_emotions_test = model_of_emotions_train.predict(x_test)
    accuracy_score_of_predictions_of_emotions_test = accuracy_score(ye_test, predictions_of_emotions_test)

    classifier_of_sentiments_train = MLPClassifier(activation='logistic', max_iter=2)
    model_of_sentiments_train = classifier_of_sentiments_train.fit(x_train, ys_train)
    predictions_of_sentiments_test = model_of_sentiments_train.predict(x_test)
    accuracy_score_of_predictions_of_sentiments_test = accuracy_score(ys_test, predictions_of_sentiments_test)

    f.write(f"====================================== BASE-MLP ======================================\n\n")
    f.write(f"Emotion Score: {accuracy_score_of_predictions_of_emotions_test}\n\n")
    f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, predictions_of_emotions_test)}\n")
    f.write(f"Sentiment Score: {accuracy_score_of_predictions_of_sentiments_test}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, predictions_of_sentiments_test)}\n")


# Part 3.6 and 3.7
def part_2_3_6(f):
  '''
    Function for Top Multi-Layered Perceptron with GridSearchCV
    ouputs TOP-MLP data: Best EmotionScore , Best SentimentScore , Classification Report
  '''

  param = {"activation": ("identity", "logistic", "tanh", "relu"), "hidden_layer_sizes": ((5, 5), (5, 10)), "solver": ("adam", "sgd")}
  model = GridSearchCV(estimator=MLPClassifier(activation='logistic', max_iter=2), param_grid=param)
  model.fit(x_train, ye_train)
  emo_predictions = model.best_estimator_.predict(x_test)
  emo_est = model.best_estimator_
  emo_score = model.best_score_

  model.fit(x_train, ys_train)
  sen_predictions = model.best_estimator_.predict(x_test)
  sen_est = model.best_estimator_
  sen_score = model.best_score_

  # For part 2.4
  f.write(f"====================================== TOP-MLP ======================================\n\n")
  f.write(f"Best Emotion Score: {emo_score}\n\n")
  f.write(f"Best Emotion Estimator: {emo_est}\n\n")
  f.write(f"Emotion Classification Report: \n{classification_report(ye_test, emo_predictions)}\n")
  f.write(f"Best Sentiment Score: {sen_score}\n\n")
  f.write(f"Best Sentiment Estimator: {sen_est}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sen_predictions)}\n")

  # Confusion Matrix - Emotions
  plt.clf()
  cme = confusion_matrix(ye_test, emotion_prediction)
  cmp = ConfusionMatrixDisplay(cme)
  fig, ax = plt.subplots(figsize=(15, 15))
  plt.title('Confusion Matrix of Emotions 2.3.6 TOP MLP')
  plt.xlabel('Predict Emotions')
  plt.ylabel('True Emotions')
  cmp.plot(ax=ax, cmap='viridis')
  plt.show()

  # Confusion Matrix - Sentiments
  plt.clf()
  cms = confusion_matrix(ys_test, sentiment_prediction)
  ConfusionMatrixDisplay(cms).plot()
  plt.title('Confusion Matrix of Sentiments 2.3.6 TOP MLP')
  plt.xlabel('Predict Sentiments')
  plt.ylabel('True Sentiments')
  plt.show()


with open(output_path, 'w') as f:
  part_3_5(f)