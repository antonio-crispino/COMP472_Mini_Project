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
