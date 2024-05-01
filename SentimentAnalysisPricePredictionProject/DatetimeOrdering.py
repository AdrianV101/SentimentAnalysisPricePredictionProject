from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# this program simply orders the tweets in ascending timestamps

df = pd.read_csv("Data/Bitcoin_ONLY_tweets_link_source_filtered_with_sentiment.csv")

df_sorted = df.sort_values(by='date',ascending=True)
df_sorted.reset_index(drop=True, inplace=True)
for i in range(len(df_sorted)-1):
    try:
        assert(df_sorted.iloc[i].date<=df_sorted.iloc[i+1].date)
    except AssertionError:
        logger.warning(f"Assert for index {i} failed")
df_sorted.to_csv('../Data/Bitcoin_ONLY_dt_ordered_tweets_link_source_filtered_with_sentiment.csv', index=False)
