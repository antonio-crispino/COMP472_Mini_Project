import gzip
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import pandas as pd

main_directory = os.path.join(os.getcwd(), 'Mini_Project_1')
dataset_path = '/Users/mairamalhi/JupyterNotebook/COMP472_Mini_Project/Mini_Project_1/Dataset/goemotions.json.gz'
output_path = '/Users/mairamalhi/JupyterNotebook/COMP472_Mini_Project/Mini_Project_1/Part_2/Output.txt'

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
def part_2_2(X):
    '''
      Function splits dataset for train and test (80% - 20%)
      :param: Independant variable
      :return: x_train, x_test, ye_train, ye_test, ys_train, ys_test
    '''

    ye = emotion  # dependent variable EMOTION
    ys = sentiment  # dependent variable SENTIMENT
    x_train, x_test, ye_train, ye_test, ys_train, ys_test = train_test_split(X, ye, ys, test_size=0.2, random_state=2)
    return x_train, x_test, ye_train, ye_test, ys_train, ys_test


# Part 2.3.1
def part_2_3_1(f):
    '''
      A Multinomial Naive Bayes Classifier (Base-MNB)
    '''

    # Max iteration chosen to be small to reduce runtime
    emotion_classifier = MultinomialNB()  # Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
    emotion_model = emotion_classifier.fit(x_train, ye_train)  # Train the model from the classifier
    emotion_prediction = emotion_model.predict(x_test)  # Perform classification on the array of test vectors
    emotion_accuracy = accuracy_score(ye_test, emotion_prediction)

    # Max iteration chosen to be small to reduce runtime
    sentiment_classifier = MultinomialNB()
    sentiment_model = sentiment_classifier.fit(x_train, ys_train)
    sentiment_prediction = sentiment_model.predict(x_test)
    sentiment_accuracy = accuracy_score(ys_test, sentiment_prediction)

    f.write(f"====================================== BASE-MNB ======================================\n\n")
    f.write(f"Emotion Score: {emotion_accuracy}\n\n")
    f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction, zero_division=1)}\n")
    f.write(f"Sentiment Score: {sentiment_accuracy}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")

    # 2.4.2
    plt.clf()
    cme = confusion_matrix(ye_test, emotion_prediction)
    cmp = ConfusionMatrixDisplay(cme)
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.title('Confusion Matrix of Emotions 2.3.1 BASE MNB')
    plt.xlabel('Predict Emotions')
    plt.ylabel('True Emotions')
    cmp.plot(ax=ax, cmap='viridis')
    plt.show()

    # 2.4.2
    plt.clf()
    cms = confusion_matrix(ys_test, sentiment_prediction)
    ConfusionMatrixDisplay(cms).plot()
    plt.title('Confusion Matrix of Sentiments 2.3.1 BASE MNB')
    plt.xlabel('Predict Sentiments')
    plt.ylabel('True Sentiments')
    plt.show()


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

    sentiment_classifier = MLPClassifier(activation='logistic', max_iter=2)
    sentiment_model = sentiment_classifier.fit(x_train, ys_train)
    sentiment_prediction = sentiment_model.predict(x_test)
    sentiment_accuracy = accuracy_score(ys_test, sentiment_prediction)

    f.write(f"====================================== BASE-MLP ======================================\n\n")
    f.write(f"Emotion Score: {emotion_accuracy}\n\n")
    f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction, zero_division=1)}\n")
    f.write(f"Sentiment Score: {sentiment_accuracy}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")

    # 2.4.2
    plt.clf()
    cme = confusion_matrix(ye_test, emotion_prediction)
    cmp = ConfusionMatrixDisplay(cme)
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.title('Confusion Matrix of Emotions 2.3.3 BASE MLP')
    plt.xlabel('Predict Emotions')
    plt.ylabel('True Emotions')
    cmp.plot(ax=ax, cmap='viridis')
    plt.show()

    # 2.4.2
    plt.clf()
    cms = confusion_matrix(ys_test, sentiment_prediction)
    ConfusionMatrixDisplay(cms).plot()
    plt.title('Confusion Matrix of Sentiments 2.3.3 BASE MLP')
    plt.xlabel('Predict Sentiments')
    plt.ylabel('True Sentiments')
    plt.show()

# Part 2.3.4
def part_2_3_4(f):
    '''
      Function for MNB with GridSearchCV
      ouputs TOP-MNB data: Best EmotionScore + Estimator, Best SentimentScore + Estimator, CLassification Report
    '''

    param = {'alpha': [0.5, 0, 1, 0.2]}
    model = GridSearchCV(estimator=MultinomialNB(), param_grid=param)
    model.fit(x_train, ye_train)
    emotion_prediction = model.best_estimator_.predict(x_test)
    emotion_best_estimator = model.best_estimator_
    emotion_score = model.best_score_

    model.fit(x_train, ys_train)
    sentiment_prediction = model.best_estimator_.predict(x_test)
    sentiment_best_estimator = model.best_estimator_
    sentiment_score = model.best_score_

    f.write(f"====================================== TOP-MNB ======================================\n\n")
    f.write(f"Best Emotion Score: {emotion_score}\n\n")
    f.write(f"Best Emotion Estimator: {emotion_best_estimator}\n\n")
    f.write(f"Emotion Classfication Report: \n{classification_report(ye_test, emotion_prediction)}\n")
    f.write(f"Best Sentiment Score: {sentiment_score}\n\n")
    f.write(f"Best Sentiment Estimator: {sentiment_best_estimator}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sentiment_prediction)}\n")

    # 2.4.2
    plt.clf()
    cme = confusion_matrix(ye_test, emotion_prediction)
    cmp = ConfusionMatrixDisplay(cme)
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.title('Confusion Matrix of Emotions 2.3.4 TOP MNB')
    plt.xlabel('Predict Emotions')
    plt.ylabel('True Emotions')
    cmp.plot(ax=ax, cmap='viridis')
    plt.show()

    # 2.4.2
    plt.clf()
    cms = confusion_matrix(ys_test, sentiment_prediction)
    ConfusionMatrixDisplay(cms).plot()
    plt.title('Confusion Matrix of Sentiments 2.3.4 TOP MNB')
    plt.xlabel('Predict Sentiments')
    plt.ylabel('True Sentiments')
    plt.show()


# Part 2.3.5
def part_2_3_5(f):
    '''
    Function for Top DT with GridSearchCV
    ouputs TOP-DT data: Best EmotionScore , Best SentimentScore , Classification Report
  '''

    param = {"criterion": ("gini", "entropy"), "max_depth": (100, 3), "min_samples_split": (12, 5, 30)}
    model = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param)
    model.fit(x_train, ye_train)
    emo_predictions = model.best_estimator_.predict(x_test)
    emo_est = model.best_estimator_
    emo_score = model.best_score_

    model.fit(x_train, ys_train)
    sen_predictions = model.best_estimator_.predict(x_test)
    sen_est = model.best_estimator_
    sen_score = model.best_score_

    f.write(f"====================================== TOP-DT ======================================\n\n")
    f.write(f"Best Emotion Score: {emo_score}\n\n")
    f.write(f"Best Emotion Estimator: {emo_est}\n\n")
    f.write(f"Emotion Classification Report: \n{classification_report(ye_test, emo_predictions)}\n")
    f.write(f"Best Sentiment Score: {sen_score}\n\n")
    f.write(f"Best Sentiment Estimator: {sen_est}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sen_predictions)}\n")

    # 2.4.2
    plt.clf()
    cme = confusion_matrix(ye_test, emotion_prediction)
    cmp = ConfusionMatrixDisplay(cme)
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.title('Confusion Matrix of Emotions 2.3.5 TOP DT')
    plt.xlabel('Predict Emotions')
    plt.ylabel('True Emotions')
    cmp.plot(ax=ax, cmap='viridis')
    plt.show()

    # 2.4.2
    plt.clf()
    cms = confusion_matrix(ys_test, sentiment_prediction)
    ConfusionMatrixDisplay(cms).plot()
    plt.title('Confusion Matrix of Sentiments 2.3.5 TOP DT')
    plt.xlabel('Predict Sentiments')
    plt.ylabel('True Sentiments')
    plt.show()

# Part 2.3.6
def part_2_3_6(f):
    '''
    Function for Top Multi-Layered Perceptron with GridSearchCV
    ouputs TOP-MLP data: Best EmotionScore , Best SentimentScore , Classification Report
  '''

    param = {"activation": ("identity", "logistic", "tanh", "relu"), "hidden_layer_sizes": ((5, 5), (5, 10)),
             "solver": ("adam", "sgd")}
    model = GridSearchCV(estimator=MLPClassifier(activation='logistic', max_iter=2), param_grid=param)
    model.fit(x_train, ye_train)
    emo_predictions = model.best_estimator_.predict(x_test)
    emo_est = model.best_estimator_
    emo_score = model.best_score_

    model.fit(x_train, ys_train)
    sen_predictions = model.best_estimator_.predict(x_test)
    sen_est = model.best_estimator_
    sen_score = model.best_score_

    f.write(f"====================================== TOP-MLP ======================================\n\n")
    f.write(f"Best Emotion Score: {emo_score}\n\n")
    f.write(f"Best Emotion Estimator: {emo_est}\n\n")
    f.write(f"Emotion Classification Report: \n{classification_report(ye_test, emo_predictions)}\n")
    f.write(f"Best Sentiment Score: {sen_score}\n\n")
    f.write(f"Best Sentiment Estimator: {sen_est}\n\n")
    f.write(f"Sentiment Classification Report \n{classification_report(ys_test, sen_predictions)}\n")

    # 2.4.2
    plt.clf()
    cme = confusion_matrix(ye_test, emotion_prediction)
    cmp = ConfusionMatrixDisplay(cme)
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.title('Confusion Matrix of Emotions 2.3.6 BASE MLP')
    plt.xlabel('Predict Emotions')
    plt.ylabel('True Emotions')
    cmp.plot(ax=ax, cmap='viridis')
    plt.show()

    # 2.4.2
    plt.clf()
    cms = confusion_matrix(ys_test, sentiment_prediction)
    ConfusionMatrixDisplay(cms).plot()
    plt.title('Confusion Matrix of Sentiments 2.3.6 BASE MLP')
    plt.xlabel('Predict Sentiments')
    plt.ylabel('True Sentiments')
    plt.show()


def part_2_5(f):
    global x_train, x_test, ye_train, ye_test, ys_train, ys_test
    vectorizer_sw_removed = CountVectorizer(stop_words='english')
    content_vector_sw_removed = vectorizer_sw_removed.fit_transform(content)
    x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2(content_vector_sw_removed)
    f.write(f"\n\n---------------------------STOP WORDS REMOVED-----------------------------------\n\n")
    part_2_3_1(f)
    part_2_3_3(f)
    part_2_3_4(f)
    part_2_3_5(f)
    part_2_3_6(f)


x_train, x_test, ye_train, ye_test, ys_train, ys_test = part_2_2(content_vector)
# Write to output file
with open(output_path, 'a+') as f:
    part_2_1(f)
    part_2_3_1(f)
    part_2_3_3(f)
    part_2_3_4(f)
    part_2_3_5(f)
    part_2_3_6(f)
    part_2_5(f)



