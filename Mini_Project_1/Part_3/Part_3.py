import os
import gzip
import json
import numpy as np
from sklearn.model_selection import train_test_split
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
print(x_train_embeddings_obj["overall_hit_rate"])
print(json.dumps(x_train_embeddings_obj[0], indent=2)) #example
print(x_test_embeddings_obj["overall_hit_rate"])
print(json.dumps(x_test_embeddings_obj[0], indent=2)) #example
# json.dump(content_embeddings_obj, open(os.path.join(main_directory, 'Part_3/embeddings', 'content_embeddings_obj.json'), 'w'), indent=2) #example
