from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import ast
import matplotlib.pyplot as plt


logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger
# this program condenses the sentiment of all the tweets in each 4 hour period, in order to make the data appropriate
# for input to a model

df=pd.read_csv("Data/Bitcoin_ONLY_dt_ordered_tweets_link_source_filtered_with_sentiment.csv", parse_dates=["date", "user_created"])
reset=False
start=True
sentiment4hr=0
sentiment1hr=0
sentiment30m=0
tweets4hr=0
neutraltweets4hr=0

data= {"start_time":[],
       "end_time":[],
       "sentiment4hr":[],
       "sentiment1hr":[],
       "sentiment30m":[],
       "tweets4hr":[],
       "neutraltweets4hr":[]}

sentimentdf=pd.DataFrame(data)

def numerical_sentiment(sentiment):
    list_dict = ast.literal_eval(sentiment)
    s_dict = list_dict[0]
    if s_dict["label"] == 'negative':
        return s_dict['score']*-1
    elif s_dict["label"] == 'neutral':
        return 0
    elif s_dict["label"] == 'positive':
        return s_dict['score'] * 1
    else:
        logger.warning("Sentiment input to numerical_sentiment function is not a case")
        return 0

for index, row in df.iterrows():
    rtime=row["date"]
    if start:
        start_time=rtime
        end_time=start_time+pd.Timedelta(hours=4)
        start=False
        reset=False
    if reset:
        start_time=end_time
        end_time=start_time+pd.Timedelta(hours=4)
        reset=False
    if rtime>=start_time and rtime<end_time:
        sentiment4hr+=numerical_sentiment(row["Sentiment"])
        tweets4hr+=1
        if numerical_sentiment(row["Sentiment"])==0:
            neutraltweets4hr+=1
    if rtime>=start_time+pd.Timedelta(hours=3) and rtime<end_time:
        sentiment1hr+=numerical_sentiment(row["Sentiment"])
    if rtime>=start_time+pd.Timedelta(hours=3, minutes=30) and rtime<end_time:
        sentiment30m+=numerical_sentiment(row["Sentiment"])
    elif rtime<start_time or rtime>=end_time:
        logger.info(f"For time period from {start_time} to {end_time} normalised sentiments were:")
        sentiment4hr=sentiment4hr/8
        sentiment1hr=sentiment1hr/2
        logger.info(sentiment4hr)
        logger.info(sentiment1hr)
        logger.info(sentiment30m)
        logger.info(f"{tweets4hr} tweets, {neutraltweets4hr} were neutral")
        if tweets4hr>0:
            sentimentdf.loc[len(sentimentdf)] = [start_time,end_time,sentiment4hr,sentiment1hr,sentiment30m,tweets4hr, neutraltweets4hr]
        reset = True
        sentiment4hr = 0
        sentiment1hr = 0
        sentiment30m = 0
        neutraltweets4hr=0
        tweets4hr=0

sentimentdf.to_csv("../Data/CompressedSentimentForInput.csv", header=True, index=False)


