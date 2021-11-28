import tweepy # pip install tweepy / brew install tweepy (if on a mac... probably)
#import pandas # pip install pandas / brew install pandas (if on a mac... probably)
import re

# set up and read in keys
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''
try:
    with open("keys.txt","r") as f:
        keys = f.readlines()
        CONSUMER_KEY = keys[0].rstrip()
        CONSUMER_SECRET = keys[1].rstrip()
        ACCESS_TOKEN = keys[2].rstrip()
        ACCESS_TOKEN_SECRET = keys[3].rstrip()
except IOError:
    print("make sure that keys.txt exists in the same directory as this file.")

# authenticate
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

# get the tweets. returns a list of JSON formatted strings. I think...
def get_tweets(keyword="cpec", date_since="2013-01-01", max_tweets="100"):
    tweets = tweepy.Cursor(api.search,
                        keyword, #contains the keyword
                        geocode="20.5937,78.9629,1000km", #from India
                        lang="en", #english
                        since=date_since, #tweeted after date
                        tweet_mode="extended" #gets the full tweet. If this isnt set, change tweet.full_text to tweet.text.
                        ).items(max_tweets) #only get max_tweets number of tweets. Omit for unlimited tweets.
    return tweets

# takes a list of JSON formatted strings and returns a list of just the tweet text
def strip_tweets(tweets):
    tweet_list = []
    # only get the text of the tweet and store in tweet_list
    i=1
    for tweet in tweets:
        tweet_list.append(tweet.full_text)
        # log
        print("gotten", i, "tweets so far!")
        #print(i, tweet.full_text)
        i+=1
    return tweet_list

# fix up some of the punctuation and whitespaces to make storage easier/csv format possible.
# tweets definitely need a lot more cleaning but this shouldn't hurt the data too badly.
def fix_tweets(tweet_list):
    fixed_tweets = []
    for tweet in tweet_list:
        s = tweet
        s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE) # removes links! (yoinked from stack overflow)
        s = s.replace('\n', ' ').replace(',', ' ').replace('.',' ') #ew. (replaced punctation with spaces to avoid weird conglomerations of words)
        fixed_tweets.append(s)
    return fixed_tweets

# store the tweets into a CSV file named "tweets.csv" that can be read / analyzed later on!
# currently APPENDS to the .csv file. this can be easily changed to write by modifying the open() parameters!
def store_tweets(tweet_list:list):
    with open('tweets2.csv', 'a', encoding="utf-8") as file: #need to specify utf-8 encoding for some reason. Dont ask me why.
        for tweet_text in tweet_list:
            #print(tweet_text)
            file.write(tweet_text+",")
    file.close()

# get, strip, clean, and store!
def main():
    tweets = get_tweets("cpec", "2013-01-01", 100)
    tweet_list = strip_tweets(tweets)
    tweet_list = fix_tweets(tweet_list)
    store_tweets(tweet_list)

if __name__ == "__main__":
    main()