from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import time

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# this program was made to see what range of times the tweets covered

chunk_size=5000
chunk_counter=0
first_pass= True
for chunk in pd.read_csv('Data/Bitcoin_tweets_link_source_filtered.csv', chunksize=chunk_size): #dtype=dtype_dictionary):
    start_time = time.time()
    if first_pass:
        earliest_date = chunk['date'].min()
        latest_date = chunk['date'].max()
        first_pass=False
    else:
        if chunk['date'].min() < earliest_date:
            earliest_date = chunk['date'].min()
        if chunk['date'].max() > latest_date:
            latest_date = chunk['date'].max()
    chunk_counter+=1
    end_time=time.time()
    logger.info(f"Have completed {chunk_counter} chunks. Took {end_time-start_time} seconds to process 5000 tweets, so expecting total runtime of {655*(end_time-start_time)/(5*3600)} hours")
logger.info(f"Earliest date is {earliest_date}, latest date is {latest_date}")