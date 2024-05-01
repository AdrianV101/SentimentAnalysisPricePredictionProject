from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import time
import re


logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# This script filters out all tweets that do not explicitly discuss Bitcoin, and writes the remaining tweets to a new file

chunk_size = 5000
chunk_counter = 0
first_pass = True

for chunk in pd.read_csv('Data/Bitcoin_tweets_link_source_filtered_with_sentiment.csv', chunksize=chunk_size, encoding="utf-8"):
    start_time = time.time()
    chunk_counter += 1

    filtered_rows = []

    for index, row in chunk.iterrows():
        if (" bitcoin" in row["text"].lower()) or (" btc" in row["text"].lower()):
            filtered_rows.append(row)
    filtered_chunk = pd.DataFrame(filtered_rows)
    if first_pass:
        filtered_chunk.to_csv("../Data/Bitcoin_ONLY_tweets_link_source_filtered_with_sentiment.csv", mode="w", header=True, index=False)
        first_pass = False
    else:
        filtered_chunk.to_csv("../Data/Bitcoin_ONLY_tweets_link_source_filtered_with_sentiment.csv", mode="a", header=False, index=False)
    end_time = time.time()
    logger.info(f"Have completed {chunk_counter} chunks. Took {end_time-start_time} seconds to process 5000 tweets, so expecting total runtime of {1600*(end_time-start_time)/(5*3600)} hours")