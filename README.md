# twitter_harassment
Analyzing Harassment on Twitter

The src directory contains the ipython notebooks used in this research. The functionality of important files are as follows:
- data_collection.ipynb: Code used to download data from the Twitter API using the tweepy library.
- classifier_utils.py: The python file which implements the classifier. The `get_pipeline` method returns a classifier (An SciKit Learn Pipeline) which takes as input a list of strings (Tweets) and labels.
- train_classifiers.ipynb: Trains the classifier on the four datasets, evaluates the performance and saves the trained model on the disk to be used later.
- predict_blocklists.ipynb: Runs the classifier on tweets from users of different blocklists and produces the histograms.
- predict_victims.ipynb: Runs the classifier on tweets that mention potential victims (female senetors and journalists) and produces the histograms.
- predict_harassers.ipynb: Identify users who harass the above victims, and analyze the rest of their tweets.
- dempgraphic_prediction.ipynb: Use the dempgraphic prediction models to predict demographics of harassers, and run the regression analysis.

For more details refer paper.pdf
