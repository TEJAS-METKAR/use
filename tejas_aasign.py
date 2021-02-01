from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
print(module_url)
model = hub.load(module_url)


print ("module %s loaded" % module_url)
def embed(input):
    return model(input)

sen1= "outdat inform credit report previou disput yet..."
sentence = " purcha new car xxxx xxxx car dealer call citiz...."
paragraph = ("account credit report mistaken date mail debt ...")
messages = [sen1, sentence, paragraph]
logging.set_verbosity(logging.ERROR)
message_embeddings = embed(messages)
for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Message: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

t=pd.read_excel('/home/ubuntu/Downloads/consumer_complaint.xlsx',engine='openpyxl')
t.head()
print(t)
td=t['Consumer_complaint_narrative']
sen1=td[0]
sen2=td[1]
sen3=td[2]
message1=[sen1,sen2,sen3]
logging.set_verbosity(logging.ERROR)
message_embeddings = embed(messages)
for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #print("Message: {}".format(messages[i]))
    #print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

from sklearn.metrics.pairwise import cosine_similarity
#
#import tensorflow as tf
#import tensorflow_hub as hub
ya=input("enter the querry:")
#score1 = []
highest_score = -1
highest_score_index = 0
def process_use_similarity():
    highest_score = -1
    highest_score_index = 0
    for p in range(0,9999):
        base_document = ya
        documents = [td[p]]
        base_embeddings = model([base_document])
        embeddings = model(documents)
        embeddings = model(documents)
        scores = cosine_similarity(base_embeddings, embeddings).flatten()
        #score1 = np.append (score1,score) 
        for i, score in enumerate(scores):
            if highest_score < score:
                highest_score = score
                highest_score_index = p
    most_similar_document = td[highest_score_index]
    print("Most similar document by USE with the score i index:",
               highest_score,highest_score_index)
    print("line",td[p])

process_use_similarity()
