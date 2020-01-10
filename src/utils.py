import pickle
import sys
import time
import glob
import re
import os.path



def load_user_tweets(directory, min_tweets=0, limit=None):
    user_tweets = {}
    i = 0
    for f in glob.glob(directory + '/*.p'):
        uid = int(re.findall(r'\d+', f.replace(directory, ''))[0])
        with open(f, 'rb') as file:
            tweets = pickle.load(file)
        texts = [t.text for t in tweets if not hasattr(t, 'retweeted_status')]
        if len(texts) > min_tweets:
            user_tweets[uid] = texts
            i += 1
        if limit is not None and i > limit:
            break
    return user_tweets


def load_users(directory, min_tweets=0, limit=None):
    users = []
    for f in glob.glob(directory + '/*.p'):
        uid = int(re.findall(r'\d+', f.replace(directory, ''))[0])
        with open(f, 'rb') as file:
            tweets = pickle.load(file)
        if len(tweets) > min_tweets:
            users.append(tweets[0].user)
            
        if limit is not None and len(users) > limit:
            break
    return users


def load_user_tweets_by_name(directory, usernames, text_only=True):
    user_tweets = {}
    for u in usernames:
        path = directory + '/' + u + '.p'
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as file:
            tweets = pickle.load(file)
        if text_only:
            texts = [t.text for t in tweets if not hasattr(t, 'retweeted_status')]
            user_tweets[u] = texts
        else:
            user_tweets[u] = [t for t in tweets if not hasattr(t, 'retweeted_status')]
    return user_tweets

def load_users_by_name(directory, usernames):
    users = []
    for u in usernames:
        path = directory + '/' + u + '.p'
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as file:
            tweets = pickle.load(file)
        if len(tweets) > 0:
            users.append(tweets[0].user)
    return users