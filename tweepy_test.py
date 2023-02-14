import tweepy
import pandas as pd
import csv
import json
import config


# Replace your bearer token below
# client = Twarc2(bearer_token="XXXXX")

# Twitter authentication
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)   
# auth.set_access_token(access_key, access_secret) 
# auth = tweepy.OAuth2BearerHandler(bearer_token)

  
# Creating an API object 
# api = tweepy.API(auth)
client = tweepy.Client(bearer_token=config.bearer_token)



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