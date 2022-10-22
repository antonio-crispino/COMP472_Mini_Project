import gzip
import json
import os
import numpy as np
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

# transform the feature array (content_array) into a feature vector (content_vector)
vectorizer = CountVectorizer()
content_vector = vectorizer.fit_transform(content_array)

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
x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2(content_vector)

# Part 2.3.1
def part_2_3_1(f):
  '''
    A Multinomial Naive Bayes Classifier (Base-MNB)
  '''

  emotion_classifier = MultinomialNB() # Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
  emotion_model = emotion_classifier.fit(x_train, ye_train) # Train the model from the classifier
  emotion_prediction = emotion_model.predict(x_test) # Perform classification on the array of test vectors
  emotion_accuracy = accuracy_score(ye_test, emotion_prediction) # Get accuracy of predictions on test set

  sentiment_classifier = MultinomialNB()
  sentiment_model = sentiment_classifier.fit(x_train, ys_train)
  sentiment_prediction = sentiment_model.predict(x_test)
  sentiment_accuracy = accuracy_score(ys_test, sentiment_prediction)

  # For part 2.4
  f.write(f"====================================== BASE-MNB ======================================\n\n")
  f.write(f"Emotion Score: {emotion_accuracy}\n\n")
  f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction, zero_division=1)}\n")
  f.write(f"Sentiment Score: {sentiment_accuracy}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")

# Part 2.3.2
def part_2_3_2(f):
  '''
    A Decision Tree Classifier (Base-DT)
  '''

  emotion_classifier = DecisionTreeClassifier()
  emotion_model = emotion_classifier.fit(x_train, ye_train)
  emotion_prediction = emotion_model.predict(x_test)
  emotion_accuracy = accuracy_score(ye_test, emotion_prediction)

  sentiment_classifier = DecisionTreeClassifier()
  sentiment_model = sentiment_classifier.fit(x_train, ys_train)
  sentiment_prediction = sentiment_model.predict(x_test)
  sentiment_accuracy = accuracy_score(ys_test, sentiment_prediction)

  # For part 2.4
  f.write(f"====================================== BASE-DT ======================================\n\n")
  f.write(f"Emotion Score: {emotion_accuracy}\n\n")
  f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction, zero_division=1)}\n")
  f.write(f"Sentiment Score: {sentiment_accuracy}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")

# Part 2.3.3
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

# Part 2.3.4
def part_2_3_4(f):
  '''
    Function for MNB with GridSearchCV
    ouputs TOP-MNB data: Best EmotionScore + Estimator, Best SentimentScore + Estimator, CLassification Report
  '''

  param = {'alpha': [0.5, 0, 1, 0.2]}
  emotion_classifier = GridSearchCV(estimator=MultinomialNB(), param_grid=param)
  emotion_model = emotion_classifier.fit(x_train, ye_train)
  emotion_prediction = emotion_model.best_estimator_.predict(x_test)
  emotion_best_estimator = emotion_model.best_estimator_
  emotion_score = emotion_model.best_score_

  sentiment_classifier = GridSearchCV(estimator=MultinomialNB(), param_grid=param)
  sentiment_model = sentiment_classifier.fit(x_train, ys_train)
  sentiment_prediction = sentiment_model.best_estimator_.predict(x_test)
  sentiment_best_estimator = sentiment_model.best_estimator_
  sentiment_score = sentiment_model.best_score_

  # For part 2.4
  f.write(f"====================================== TOP-MNB ======================================\n\n")
  f.write(f"Best Emotion Score: {emotion_score}\n\n")
  f.write(f"Best Emotion Estimator: {emotion_best_estimator}\n\n")
  f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction)}\n")
  f.write(f"Best Sentiment Score: {sentiment_score}\n\n")
  f.write(f"Best Sentiment Estimator: {sentiment_best_estimator}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")

# Part 2.3.5
def part_2_3_5(f):
  '''
    Function for Top DT with GridSearchCV
    ouputs TOP-DT data: Best EmotionScore , Best SentimentScore , Classification Report
  '''

  param = {"criterion": ("gini", "entropy"), "max_depth": (100, 3), "min_samples_split": (12, 5, 30)}
  emotion_classifier = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param)
  emotion_model = emotion_classifier.fit(x_train, ye_train)
  emo_predictions = emotion_model.best_estimator_.predict(x_test)
  emo_est = emotion_model.best_estimator_
  emo_score = emotion_model.best_score_

  sentiment_classifier = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param)
  sentiment_model = sentiment_classifier.fit(x_train, ys_train)
  sen_predictions = sentiment_model.best_estimator_.predict(x_test)
  sen_est = sentiment_model.best_estimator_
  sen_score = sentiment_model.best_score_

  # For part 2.4
  f.write(f"====================================== TOP-DT ======================================\n\n")
  f.write(f"Best Emotion Score: {emo_score}\n\n")
  f.write(f"Best Emotion Estimator: {emo_est}\n\n")
  f.write(f"Emotion Classification Report: \n{classification_report(ye_test, emo_predictions)}\n")
  f.write(f"Best Sentiment Score: {sen_score}\n\n")
  f.write(f"Best Sentiment Estimator: {sen_est}\n\n")
  f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sen_predictions)}\n")

# Part 2.3.6
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

# Part 2.5
def part_2_5(f):
  global x_train, x_test, ye_train, ye_test, ys_train, ys_test
  vectorizer_sw_removed = CountVectorizer(stop_words='english') 
  content_vector_sw_removed = vectorizer_sw_removed.fit_transform(content_array)
  x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2(content_vector_sw_removed)

  f.write(f"\n\n---------------------------STOP WORDS REMOVED-----------------------------------\n\n")
  part_2_3_1(f)
  part_2_3_2(f)
  part_2_3_3(f)
  part_2_3_4(f)
  part_2_3_5(f)
  part_2_3_6(f)
  
# Write to output file
with open(output_path, 'a+') as f:
  part_2_1(f)
  part_2_3_1(f)
  part_2_3_2(f)
  part_2_3_3(f)
  part_2_3_4(f)
  part_2_3_5(f)
  part_2_3_6(f)
  part_2_5(f)
