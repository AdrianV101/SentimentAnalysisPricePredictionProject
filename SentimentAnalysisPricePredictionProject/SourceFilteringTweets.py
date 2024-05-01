from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import time

# this program removes all the tweets that are not from the sources from which a normal (non-spam) user would tweet
logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

chunk_size = 5000
chunk_counter = 0
first_pass = True

for chunk in pd.read_csv('Data/Bitcoin_tweets.csv', chunksize=chunk_size, encoding="utf-8"):
    start_time = time.time()
    chunk_counter += 1

    accepted_sources = ['Twitter Web App','Twitter for Android', 'Twitter for iPhone', 'Twitter for iPad'] #Sources that are accepted ie. only manually inputted to twitter tweets
    filtered_chunk = chunk[chunk['source'].isin(accepted_sources)]
    if first_pass:
        filtered_chunk.to_csv("../Data/Bitcoin_tweets_source_filtered.csv", mode="w", header=True, index=False)
        first_pass = False
    else:
        filtered_chunk.to_csv("../Data/Bitcoin_tweets_source_filtered.csv", mode="a", header=False, index=False)
    end_time = time.time()
    logger.info(f"Have completed {chunk_counter} chunks. Took {end_time-start_time} seconds to process 5000 tweets, so expecting total runtime of {4600*(end_time-start_time)/(5*3600)} hours")
