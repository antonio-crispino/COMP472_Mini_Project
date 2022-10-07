import gzip
import json
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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
cv = vectorizer.fit_transform(content).toarray()
emotion = [post[1] for post in posts]
sentiment = [post[2] for post in posts]

# 2.1
def Part_2_1(outputPath):
    '''
        Function writes total vocabulary size to output file

        :outputPath: Path of output file
        :posts: json load of post documents dataset
        :return: size of the vocabulary
    '''
    try: 
        with open(outputPath, 'a+') as f:
            # Only content of each post is considered, exclding emotion/sentiment
            word_list = vectorizer.get_feature_names_out()
            vocabSize = len(word_list)
            f.write(f"SIZE OF VOCABULARY : {vocabSize}\n\n ==============================\n")      
    except: 
        print(f"Error writing to output file")
    return vocabSize

# 2.2
def Part_2_2():
    '''
        Function splits dataset for train and test (80% - 20%)
        :posts: json load of post document dataset
        :return: X_train, X_test, y_train, y_test

    '''
    # X: independent variable CONTENT vector (features = 30449)
    # y: dependent variable EMOTION, SENTIMENT

    X = cv
    y = np.array(list(zip(emotion, sentiment)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test


Part_2_1(outputPath)
X_train, X_test, y_train, y_test = Part_2_2()


    

    

