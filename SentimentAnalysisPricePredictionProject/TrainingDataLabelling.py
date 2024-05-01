from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import time

# this script was made for when I considered training my own sentiment analysis model, which would require me manually
# labelling training data. I never ended up making that model, and so this script was never used
logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# File paths
training_file = 'Data/Sentiment_training_split.csv'

# Dictionary to store labeled DataFrames
negative_list=[]
neutral_list=[]
positive_list=[]

# Load the first 50 entries from the CSV file
df = pd.read_csv(training_file, nrows=50)

# Load the remaining entries from the CSV file
remaining_df = pd.read_csv(training_file, skiprows=range(1, 51))

# Save the remaining entries back to the original file
remaining_df.to_csv(training_file, index=False)

for i in range(50):
    logger.info("Enter 1 for negative, 2 for neutral, 3 for positive")
    logger.info(df.iloc[i].text)
    resp=0
    while resp not in [1,2,3]:
        resp = int(input())
    if resp==1:
        df.iloc[i].label="negative"
    elif resp==2:
        df.iloc[i].label="neutral"
    elif resp==3:
        df.iloc[i].label="positive"

# Save the first 10,000 entries to a new file
df.to_csv("../Data/Labelled_Training_Data.csv", mode="w", header=True, index=False)
