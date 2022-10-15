import gzip
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

mainDir =  os.path.join(os.getcwd(), 'Mini_Project_1')
dataSetPath = os.path.join(mainDir, 'Dataset', 'goemotions.json.gz')
outputPath = os.path.join(mainDir, 'Part_2', 'Output.txt')

try: 
    with gzip.open(dataSetPath, 'rb') as f:
        posts = json.load(f)
except:
    print(f"Error reading dataset {dataSetPath}")

content = [post[0] for post in posts]
vectorizer = CountVectorizer()
cv = vectorizer.fit_transform(content)
emotion = [post[1] for post in posts]
sentiment = [post[2] for post in posts]

# 2.1
def Part_2_1(f):
    '''
        Function writes total vocabulary size to output file
        :return: size of the vocabulary
    '''
  
    word_list = vectorizer.get_feature_names_out()
    vocabSize = len(word_list)
    f.write(f"SIZE OF VOCABULARY : {vocabSize}\n\n")      

    return vocabSize

# 2.2
def Part_2_2():
    '''
        Function splits dataset for train and test (80% - 20%)
        :return: X_train, X_test, y_train, y_test
    '''
    # X: independent variable CONTENT vector (features = 30449)
    # y: dependent variable EMOTION, SENTIMENT

    X = cv
    yEmo = emotion
    ySen = sentiment

    X_train, X_test, yEmo_train, yEmo_test, ySen_train, ySen_test = train_test_split(X, yEmo, ySen, test_size=0.2, random_state=2)

    return X_train, X_test, yEmo_train, yEmo_test, ySen_train, ySen_test



def Part_2_3_3(f):
    '''
        Function to for MLP classification
        outputs BASE-MLP data: EmotionScore, SentimentScore, Classification Report
    '''

    # Max iteration chosen to be small to reduce runtime
    model = MLPClassifier(activation='logistic', max_iter=2)

    model.fit(X_train, yEmo_train)
    emoPredictions = model.predict(X_test)
    emoScore = accuracy_score(yEmo_test, emoPredictions)

    model.fit(X_train, ySen_train)
    senPredictions = model.predict(X_test)
    senScore = accuracy_score(ySen_test, senPredictions)

    f.write(f"================ BASE-MLP ===================\nEmotion Score: {emoScore}\n")
    f.write(f"Classfication Report: \n{classification_report(yEmo_test, emoPredictions)}\n\n")
    f.write(f"Sentiment Score: {senScore}\n")
    f.write(f"Classification Report \n{classification_report(ySen_test, senPredictions)}\n\n")
    
    # 2.4.2
    # cm = confusion_matrix(yEmo_test, emoPredictions)
    # ConfusionMatrixDisplay(cm).plot()
    # plt.show()
   
def Part_2_3_4(f):
    '''
        Function for MNB with GridSearchCV
        ouputs TOP-MNB data: Best EmotionScore + Estimator, Best SentimentScore + Estimator, CLassification Report
    '''
    param = {"alpha": [0.5, 0, 1, 0.2]}
    model = GridSearchCV(estimator=MultinomialNB(), param_grid=param)
    model.fit(X_train, yEmo_train)
    emoPredictions = model.best_estimator_.predict(X_test)
    emoEst = model.best_estimator_
    emoScore = model.best_score_

    model.fit(X_train, ySen_train)
    senPredictions = model.best_estimator_.predict(X_test)
    senEst = model.best_estimator_
    senScore = model.best_score_

    f.write(f"================ TOP-MNB ===================\nBest Emotion Score: {emoScore}, Best Emotion Estimator: {emoEst}\n")
    f.write(f"Classfication Report: \n{classification_report(yEmo_test, emoPredictions)}\n\n")
    f.write(f"Best Sentiment Score: {senScore}, Best Sentiment Estimator: {senEst}\n")
    f.write(f"Classification Report \n{classification_report(ySen_test, senPredictions)}\n\n")

with open(outputPath, 'a+') as f:
    Part_2_1(f)
    X_train, X_test, yEmo_train, yEmo_test, ySen_train, ySen_test = Part_2_2()
    Part_2_3_3(f)
    Part_2_3_4(f)
    

    

