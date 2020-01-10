from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from textstat.textstat import *
import nltk
from nltk.stem.porter import *
import string
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
import brownclusters as bc

bc.load()

tknzr = TweetTokenizer(reduce_len=True)

class OtherFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.sentiment_analyzer = VS()
    
    def count_twitter_objs(self, text_string):
        """
        Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE
        4) hashtags with HASHTAGHERE

        This allows us to get standardized counts of urls and mentions
        Without caring about specific people mentioned.

        Returns counts of urls, mentions, and hashtags.
        """
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        hashtag_regex = '#[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
        parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
        parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
        return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

    def other_features(self, tweet):
        """This function takes a string and returns a list of features.
        These include Sentiment scores, Text and Readability scores,
        as well as Twitter specific features"""
        ##SENTIMENT
        sentiment = self.sentiment_analyzer.polarity_scores(tweet)

        words = preprocess(tweet) #Get text only

        syllables = textstat.syllable_count(words) #count syllables in words
        num_chars = sum(len(w) for w in words) #num chars in words
        num_chars_total = len(tweet)
        num_terms = len(tweet.split())
        num_words = len(words.split())
        avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
        num_unique_terms = len(set(words.split()))

        ###Modified FK grade, where avg words per sentence is just num words/1
        FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
        ##Modified FRE score, where sentence fixed to 1
        FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

        twitter_objs = self.count_twitter_objs(tweet) #Count #, @, and http://
        retweet = 0
        if "rt" in words:
            retweet = 1
        features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                    num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                    twitter_objs[2], twitter_objs[1],
                    twitter_objs[0], retweet]
        #features = pandas.DataFrame(features)
        return features

    def fit(self, x, y=None):
        return self

    def transform(self, recs):
        return np.array([self.other_features(t) for t in recs])
    
    def get_feature_names(self):
        return ['FKRA', 'FRE', 'syllables_count', 'avg_syl', 'num_chars', 'num_chars_total', 'num_terms', 'num_words', 'num_unique_terms', 'sentiment_neg', 'sentiment_pos', 'sentiment_neu', 'sentiment_compound', 'hashtag_count', 'mention_count', 'url_count', 'retweet', ]


stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("english")
stopwords = [stemmer.stem(t) for t in stopwords]
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URL', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTION', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

def tokenize(tweet):
    tokens = [stemmer.stem(t) for t in tknzr.tokenize(tweet)]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

def pos_tokenize(tweet):
    tokens = tknzr.tokenize(tweet)
    return [x[1] for x in nltk.pos_tag(tokens)]

     

vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords, #We do better when we keep stopwords
    use_idf=True,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=20000,
    min_df=5,
    max_df=0.8
)

pos_vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=pos_tokenize,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None, #We do better when we keep stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.8,
)

bc_vectorizer = TfidfVectorizer(
    #vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    tokenizer=bc.tokenize_and_tag,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None, #We do better when we keep stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies l2 norm smoothing
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.8,
)

def get_feature_transformers():
    return  Pipeline([
            ('features', FeatureUnion([
                ('text_tfidf', vectorizer),
                ('pos_tfidf', pos_vectorizer),
                ('bc_tfidf', bc_vectorizer),
                ('other_features', OtherFeatures()),
            ])),
            ('scaler', StandardScaler(with_mean=False)),
            #('select', SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01), threshold=-np.inf, max_features=2000))
        ])
        

classifiers = {
            
            'lsvc': CalibratedClassifierCV(LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge')),
            'gboost': GradientBoostingClassifier(n_estimators=100, max_features=0.9),
            'rf': RandomForestClassifier(n_estimators=100),
            'svc': SVC(probability=True)
}


def get_pipeline(cname):
    

    pipeline = Pipeline([
        ('features', get_feature_transformers()),
        ('clf', classifiers[cname])
    ])
    return pipeline

