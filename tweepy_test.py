import tweepy
import pandas as pd
import config
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    query = 'FastAndFurious'
    i = 0
    # with open(file_name, 'a+', encoding="utf-8") as filehandle:
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, 
                                  tweet_fields=['context_annotations', 'created_at'], max_results=10).flatten(limit=1):
        i = i +1
        print(tweet.text)
        word_tokens = word_tokenize(tweet.text)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        print(filtered_sentence)
        file_name = "tweets_"+str(i)+".txt"
        File_object = open(file_name, "a+", encoding="utf-8")
        File_object.write('%s\n' % tweet.text)

    # for tweet in tweets.data:
    #     print(tweets.data)
    #     text = tweet.text
    #     if len(tweet.context_annotations) > 0:
    #         # print(len(tweet.context_annotations))
    #         refined_tweet = {
    #             'text' : text,
    #             # 'id' : tweet.context_annotations[0]['domain']['id'],
    #             # 'name' : tweet.context_annotations[0]['domain']['name'],
    #             'context' : tweet.context_annotations
    #         }
    #         list.append(refined_tweet)
    #         # print(text)
    #         # print(tweet.context_annotations)
    # print(list)
            # break
            # print(tweet.context_annotations[0]['domain']['id'])
            # jdump = json.dumps(tweet.context_annotations)
            # print(tweet.context_annotations[1])
            # out.writerow(tweet.context_annotations)
    # df = pd.DataFrame(list)
    # df.to_csv('refined_tweets.csv')


###################### Elevated access needed #########################
    # searched_tweets = [status for status in tweepy.Cursor(api.search_tweets, q=query).items(max_tweets)]
    # for tweet in searched_tweets:
    #     username = tweet.user.screen_name
    #     description = tweet.user.description
    #     location = tweet.user.location
    #     following = tweet.user.friends_count
    #     followers = tweet.user.followers_count
    #     totaltweets = tweet.user.statuses_count
    #     retweetcount = tweet.retweet_count
    #     hashtags = tweet.entities['hashtags']
    #     print(tweet)
    #     break;

#############################################################################


if __name__ == "__main__":
    main()