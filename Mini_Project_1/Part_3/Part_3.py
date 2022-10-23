import os
import gzip
import json
import numpy as np
from gensim.downloader import load
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from IPython.display import display

# declare os paths
main_directory = os.path.join(os.getcwd(), 'Mini_Project_1')
dataset_path = '/Users/mairamalhi/JupyterNotebook/COMP472_Mini_Project/Mini_Project_1/Dataset/goemotions.json.gz'

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

# Two arrays, one for vectorized
vecter_list=[] # will return embeddings for each word as an array one after the other
word_filtered_list=[] # check if words exist in other posts (otherwise skip), if so, returns same value as redditpostsample

vecter_list=[pretrained_embedding_model[word] for word in redditpostsample if word in pretrained_embedding_model.index_to_key]
word_filtered_list=[word for word in redditpostsample if word in pretrained_embedding_model.index_to_key]

# print(vecter_list) #TESTING
# print(word_filtered_list) #TESTING

# Create a data frame (puts data in clean table) using pandas
data_frame = pd.DataFrame.from_dict(dict(zip(word_filtered_list,vecter_list)),orient='index')

# Write the new dataframe file to a .json
json.dump(data_frame.to_dict(), open("embeddings_of_post.json", 'w'))
json_embedded_posts = pd.read_json("embeddings_of_post.json")
dict_tokens = json_embedded_posts.to_dict()
json_embedded_posts.head(3)

# Display the data in console
# display(data_frame) #TESTING