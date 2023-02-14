import tweepy
import pandas as pd
import csv
import json

# Replace your bearer token below
# client = Twarc2(bearer_token="XXXXX")
consumer_key = "BLUeTUU66lIhF8yAtP7cxZBdd"
consumer_secret = "0fUUcxBM6PyVforfMP0cxAEMRw7X8vhaRL5QsOFYlLhtNtBEL3" 
access_key = "BLUeTUU66lIhF8yAtP7cxZBdd"
access_secret = "0fUUcxBM6PyVforfMP0cxAEMRw7X8vhaRL5QsOFYlLhtNtBEL3"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAA%2BilgEAAAAAaOslXG%2BGux0Qn0QsnGQaQ3D1x2M%3DHDZ7DUgsk2Iw5SXUqZ6hnBzMbiFdIHpAr2NRWBvLNIpLB5iK5U"
# Twitter authentication
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)   
# auth.set_access_token(access_key, access_secret) 
auth = tweepy.OAuth2BearerHandler(bearer_token)

  
# Creating an API object 
api = tweepy.API(auth)
client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAA%2BilgEAAAAAaOslXG%2BGux0Qn0QsnGQaQ3D1x2M%3DHDZ7DUgsk2Iw5SXUqZ6hnBzMbiFdIHpAr2NRWBvLNIpLB5iK5U')



def main():
    list=[]
    query = '#SidKiaraWedding -is:retweet lang:en'
    tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)
    # hashtag_tweets = tweepy.Cursor(api.search_tweets, q="#VaccinationDrive", tweet_mode='extended').items(5)

    # out = csv.writer(open("myfile.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
    for tweet in tweets.data:
        print(tweets.data)
        text = tweet.text
        if len(tweet.context_annotations) > 0:
            # print(len(tweet.context_annotations))
            refined_tweet = {
                'text' : text,
                # 'id' : tweet.context_annotations[0]['domain']['id'],
                # 'name' : tweet.context_annotations[0]['domain']['name'],
                'context' : tweet.context_annotations
            }
            list.append(refined_tweet)
            # print(text)
            # print(tweet.context_annotations)
            # print(list)
            break
            # print(tweet.context_annotations[0]['domain']['id'])
            # jdump = json.dumps(tweet.context_annotations)
            # print(tweet.context_annotations[1])
            # out.writerow(tweet.context_annotations)
    # df = pd.DataFrame(list)
    # df.to_csv('refined_tweets.csv')



if __name__ == "__main__":
    main()