import os
import gzip
import json
import numpy as np
from gensim.downloader import load
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

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
