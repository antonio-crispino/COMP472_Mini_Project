import gzip
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

main_directory =  os.path.join(os.getcwd(), 'Mini_Project_1')
dataset_path = os.path.join(main_directory, 'Dataset', 'goemotions.json.gz')
output_path = os.path.join(main_directory, 'Part_2', 'Output.txt')

posts = []
with gzip.open(dataset_path, 'rb') as f:
  posts = json.load(f)

content = np.array([post[0] for post in posts])
emotion = np.array([post[1] for post in posts])
sentiment = np.array([post[2] for post in posts])

vectorizer = CountVectorizer()
content_vector = vectorizer.fit_transform(content)

# Part 2.1
def part_2_1(f):
  '''
    Function writes total vocabulary size to output file
    :return: size of the vocabulary
  '''
  vocabulary_list = vectorizer.get_feature_names_out()
  vocabulary_size = len(vocabulary_list)
  f.write(f"SIZE OF VOCABULARY : {vocabulary_size}\n\n")    
  return vocabulary_size

# Part 2.2
def part_2_2():
  '''
    Function splits dataset for train and test (80% - 20%)
    :return: x_train, x_test, ye_train, ye_test, ys_train, ys_test
  '''
  X = content_vector # independent variable CONTENT vector
  ye = emotion # dependent variable EMOTION
  ys = sentiment # dependent variable SENTIMENT
  x_train, x_test, ye_train, ye_test, ys_train, ys_test = train_test_split(X, ye, ys, test_size=0.2, random_state=2)
  return x_train, x_test, ye_train, ye_test, ys_train, ys_test

# Part 2.3.1
def part_2_3_1(f):
  '''
    A Multinomial Naive Bayes Classifier (Base-MNB)
  '''
  x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2()

  # Max iteration chosen to be small to reduce runtime
  classifier_of_emotions_train = MultinomialNB() # Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
  model_of_emotions_train = classifier_of_emotions_train.fit(x_train, ye_train) # Train the model from the classifier
  predictions_of_emotions_test = model_of_emotions_train.predict(x_test) # Perform classification on the array of test vectors
  accuracy_score_of_predictions_of_emotions_test = accuracy_score(ye_test, predictions_of_emotions_test)

  # Max iteration chosen to be small to reduce runtime
  classifier_of_sentiments_train = MultinomialNB()
  model_of_sentiments_train = classifier_of_sentiments_train.fit(x_train, ys_train)
  predictions_of_sentiments_test = model_of_sentiments_train.predict(x_test)
  accuracy_score_of_predictions_of_sentiments_test = accuracy_score(ys_test, predictions_of_sentiments_test)

  f.write(f"====================================== BASE-MNB ======================================\n\n")
  f.write(f"Emotion Score: {accuracy_score_of_predictions_of_emotions_test}\n\n")
  f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, predictions_of_emotions_test, zero_division=1)}\n")
  f.write(f"Sentiment Score: {accuracy_score_of_predictions_of_sentiments_test}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, predictions_of_sentiments_test)}\n")

# Part 2.3.3
def part_2_3_3(f):
  '''
    Function to for MLP classification
    outputs BASE-MLP data: EmotionScore, SentimentScore, Classification Report
  '''
  x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2()

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

  # 2.4.2
  # cm = confusion_matrix(yEmo_test, emoPredictions)
  # ConfusionMatrixDisplay(cm).plot()
  # plt.show()
   
# Part 2.3.4
def part_2_3_4(f):
  '''
    Function for MNB with GridSearchCV
    ouputs TOP-MNB data: Best EmotionScore + Estimator, Best SentimentScore + Estimator, CLassification Report
  '''
  x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2()

  param = {"alpha": [0.5, 0, 1, 0.2]}
  model = GridSearchCV(estimator=MultinomialNB(), param_grid=param)
  model.fit(x_train, ye_train)
  emo_predictions = model.best_estimator_.predict(x_test)
  emo_est = model.best_estimator_
  emo_score = model.best_score_

  model.fit(x_train, ys_train)
  sen_predictions = model.best_estimator_.predict(x_test)
  sen_est = model.best_estimator_
  sen_score = model.best_score_

  f.write(f"====================================== TOP-MNB ======================================\n\n")
  f.write(f"Best Emotion Score: {emo_score}\n\n")
  f.write(f"Best Emotion Estimator: {emo_est}\n\n")
  f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emo_predictions)}\n")
  f.write(f"Best Sentiment Score: {sen_score}\n\n")
  f.write(f"Best Sentiment Estimator: {sen_est}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sen_predictions)}\n")

# Write to output file
with open(output_path, 'w') as f:
  part_2_1(f)
  part_2_3_1(f)
  part_2_3_3(f)
  part_2_3_4(f)
