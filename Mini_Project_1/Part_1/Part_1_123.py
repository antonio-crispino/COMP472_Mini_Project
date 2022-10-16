import gzip
import json
import matplotlib.pyplot as plt
import os

posts = []
posts_count = 0
emotions = {}
sentiments = {}

main_directory =  os.path.join(os.getcwd(), 'Mini_Project_1')
dataset_path = os.path.join(main_directory, 'Dataset', 'goemotions.json.gz')
figures_path = os.path.join(main_directory, 'Part_1', 'Figures')

with gzip.open(dataset_path, 'rb') as f:
  # "posts" is an array of posts, where each post is an array 
  # that contains a post's text, its emotion, and its sentiment.
  posts = json.load(f)

  for post in posts:
    posts_count += 1
    emotion = post[1]
    sentiment = post[2]

    if emotion not in emotions:
      emotions[emotion] = 1
    else:
      emotions[emotion] += 1
    
    if sentiment not in sentiments:
      sentiments[sentiment] = 1
    else:
      sentiments[sentiment] += 1

for emotion, occurance in emotions.items():
  emotions[emotion] = {"occurance": occurance, "frequency": (occurance/posts_count)*100}

for sentiment, occurance in sentiments.items():
  sentiments[sentiment] = {"occurance": occurance, "frequency": (occurance/posts_count)*100}

# emotions bar graph (occurances)
plt.clf()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.30)
plt.bar(emotions.keys(), list(pair["occurance"] for pair in emotions.values()))
plt.title('Occurences of Emotions')
plt.xlabel('Emotions')
plt.ylabel('Occurences')
plt.savefig(os.path.join(figures_path, 'Occurences_of_Emotions.pdf'))
print("Created: Occurences_of_Emotions.pdf")

# emotions bar graph (frequencies)
plt.clf()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.30)
plt.bar(emotions.keys(), list(pair["frequency"] for pair in emotions.values()))
plt.title('Frequencies of Emotions')
plt.xlabel('Emotions')
plt.ylabel('Frequency (%)')
plt.savefig(os.path.join(figures_path, 'Frequency_of_Emotions.pdf'))
print("Created: Frequency_of_Emotions.pdf")

# sentiments bar graph (occurances)
plt.clf()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.30)
plt.bar(sentiments.keys(), list(pair["occurance"] for pair in sentiments.values()))
plt.title('Occurences of Sentiments')
plt.xlabel('Sentiments')
plt.ylabel('Occurences')
plt.savefig(os.path.join(figures_path,'Occurences_of_Sentiments.pdf'))
print("Created: Occurences_of_Sentiments.pdf")

# sentiments bar graph (frequencies)
plt.clf()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.30)
plt.bar(sentiments.keys(), list(pair["frequency"] for pair in sentiments.values()))
plt.title('Frequencies of Sentiments')
plt.xlabel('Sentiments')
plt.ylabel('Frequency (%)')
plt.savefig(os.path.join(figures_path,'Frequency_of_Sentiments.pdf'))
print("Created: Frequency_of_Sentiments.pdf")
