import tweepy
import pandas as pd
import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
string.punctuation
import preprocessor as prep
import os
import sys

# Twitter authentication 
auth = tweepy.OAuth2BearerHandler(bearer_token=config.bearer_token)

  
# Creating an API object 
api = tweepy.API(auth)
client = tweepy.Client(bearer_token=config.bearer_token)

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def main():
    dataset = sys.argv[1]
    if dataset == 'test':
        test_dataset_genration()
    elif dataset == 'training':
        reading_dataset()

def test_dataset_genration():
    query = 'FastAndFurious lang:en'
    i = 0
    
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, 
                                  tweet_fields=['context_annotations', 'created_at'], max_results=10).flatten(limit=20):
        i = i +1
        print("------------Original Text --------------------")
        print(tweet.text)
        print("------------After text cleaning---------------")
        cleaned_text = text_cleaning(tweet.text)
        print(cleaned_text)
        print("----------After removing punctuation-----------")
        punctuationfree_text = punctuation_remove(cleaned_text)
        print(punctuationfree_text)
        print("---------After removing stopwords--------------")
        stopwords_removed = stopwords_remove(punctuationfree_text)
        print(stopwords_removed)

        file_name = "tweets_"+str(i)+".txt"
        File_object = open(file_name, "a+", encoding="utf-8")
        File_object.write('%s\n' % tweet.text)

def reading_dataset():
    cwd = os.getcwd()
    path = cwd+'\dataset'
    dir_list = os.listdir(path)
    print(dir_list)
    for i in dir_list:
        with open('dataset\/'+i) as f:
            print ("\n\n***********For file "+i)
            lines = f.readlines()
            print("------------Original Text --------------------")
            print(lines[0])
            print("------------After text cleaning---------------")
            cleaned_text = text_cleaning(lines[0])
            print(cleaned_text)
            print("----------After removing punctuation-----------")
            punctuationfree_text = punctuation_remove(cleaned_text)
            print(punctuationfree_text)
            print("---------After removing stopwords--------------")
            stopwords_removed = stopwords_remove(punctuationfree_text)
            print(stopwords_removed)

def punctuation_remove(row_text):
    punctuationfree="".join([i for i in row_text if i not in string.punctuation])
    return punctuationfree.lower()

def text_cleaning(row_text):
    return prep.clean(row_text)

def stopwords_remove(row_text):
    word_tokens = word_tokenize(row_text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return filtered_sentence

if __name__ == "__main__":
    main()