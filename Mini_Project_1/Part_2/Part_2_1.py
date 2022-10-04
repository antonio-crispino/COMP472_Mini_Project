import gzip
import json
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

mainDir =  os.path.join(os.getcwd(), 'Mini_Project_1')
dataSetPath = os.path.join(mainDir, 'Dataset', 'goemotions.json.gz')
vocabPath = os.path.join(mainDir, 'Part_2', 'Vocabulary_Size&Frequency.json')

try: 
    with gzip.open(dataSetPath, 'rb') as f:
        posts = json.load(f)
except:
    print(f"Error reading dataset {dataSetPath}")
      
# 2.1
def Part_2_1(vocabPath, posts):
    '''
        Function writes total vocabulary size and their frequency (entire dataset)
        format: size of vocabulary: size
                'word': frequency

        :vocabPath: Path of output file
        :posts: json load of post documents dataset
        :return: size of the vocabulary
    '''
    try: 
        with open(vocabPath, 'a+') as vocab:
            # Only content of each post is considered, exclding emotion/sentiment
            vocabData = [post[0] for post in posts]
            vectorizer = CountVectorizer()
            cvFit = vectorizer.fit_transform(vocabData)
            word_list = vectorizer.get_feature_names_out()
            count_list = cvFit.toarray().sum(axis =0)

            jsonVocab = json.dumps(dict(zip(word_list,count_list.tolist())), indent =4)
            vocabSize = len(word_list)
            vocab.write(f"SIZE OF VOCABULARY ==> {vocabSize}\n\n")
            vocab.write(jsonVocab)        
    except: 
        print(f"Error writing to Vocabulary files")
    return vocabSize

# 2.2
def Part_2_2(posts):
    '''
        Function splits dataset for train and test (80% - 20%)

        :posts: json load of post document dataset
        :return: x_train, x_test, y_train, y_test

    '''
    content = [post[0] for post in posts]
    emotion = [post[1] for post in posts]
    sentiment = [post[2] for post in posts]
    # x: independent variable CONTENT
    # y: dependent variable EMOTION, SENTIMENT
    x = content
    y = np.array(list(zip(emotion, sentiment)))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


#Part_2_1(vocabPath, posts)
#Part_2_2(posts)

    

    

