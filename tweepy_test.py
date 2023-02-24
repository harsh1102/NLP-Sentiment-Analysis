import tweepy
import pandas as pd
import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
string.punctuation
import preprocessor as prep

# Twitter authentication 
auth = tweepy.OAuth2BearerHandler(bearer_token=config.bearer_token)

  
# Creating an API object 
api = tweepy.API(auth)
client = tweepy.Client(bearer_token=config.bearer_token)

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = set(stopwords.words('english'))



def main():
    # query = '#SidKiaraWedding -is:retweet lang:en'
    query = 'FastAndFurious lang:en'
    # query = 'Dhoni lang:en'
    i = 0
    # with open(file_name, 'a+', encoding="utf-8") as filehandle:
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, 
                                  tweet_fields=['context_annotations', 'created_at'], max_results=10).flatten(limit=1):
        i = i +1
        # test_text = '''RT @SushantNMehta: If MS Dhoni wins #IPL2023 would he go down in history as the best cricketing mind of all time? 
#MSDhoni #SushantMehta'''
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

        # file_name = "tweets_"+str(i)+".txt"
        # File_object = open(file_name, "a+", encoding="utf-8")
        # File_object.write('%s\n' % tweet.text)


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