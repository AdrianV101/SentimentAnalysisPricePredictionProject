from utils.customLogger import setup_logger
from pathlib import Path
from transformers import pipeline
import pandas as pd
import time
import torch

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# this is the primary sentiment analysis code. As can be seen below, I utilised the ROBERTA sentiment analysis model
# found on Hugging Face. I struggled to decide between using FINBERT or this, however given the often casual nature
# with which people discuss crypto on twitter, I decided against using FINBERT as it was primarily trained on formal
# financial documents, with very different language to that use on twitter.

#The code analyses the tweets in chunks of 5000, because the file containing all the tweets was too large for my laptop
# to handle. This came with the useful side effect of saving progress bit by bit, meaning if the code crashed all the
# progress would not be lost
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create the pipeline object using a twitter sentiment analysis model sourced from hugging face
find_sentiment = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)


chunk_size = 5000  # specifying chunk size because the csv file is too big to load in one go

chunk_counter=0
pd.set_option('display.max_columns', None)
# Iterate over the CSV file in chunks

'''dtype_dictionary={
    "user_name":str,
    "user_location":str,
    "user_description":str,
    "user_created":str,
    "user_followers":str,
    "user_friends":str,
    "user_favourites":str,
    "user_verified":str,
    "date":str,
    "text":str,
    "hashtags":str,
    "source":str,
    "is_retweet":str}
logger.info(dtype_dictionary)'''

first_pass=True

for chunk in pd.read_csv('Data/Bitcoin_tweets_link_source_filtered.csv', chunksize=chunk_size): #dtype=dtype_dictionary):
    start_time = time.time()
    try:
        chunk['Sentiment'] = chunk['text'].map(find_sentiment)
    except:
        logger.warning(f"There was an issue sentiment analysing at least some tweets in chunk {chunk_counter}, skipping, chunk will be omitted from data")
        continue
    chunk_counter += 1
    if first_pass:
        chunk.to_csv("../Data/Bitcoin_tweets_link_source_filtered_with_sentiment.csv", mode="w", header=True, index=False)
        first_pass=False
    else:
        chunk.to_csv("../Data/Bitcoin_tweets_link_source_filtered_with_sentiment.csv", mode="a", header=False, index=False)
    end_time=time.time()
    logger.info(f"Have completed {chunk_counter} chunks. Took {end_time-start_time} seconds to process 5000 tweets, so expecting total runtime of {1730*(end_time-start_time)/(10*3600)} hours")


