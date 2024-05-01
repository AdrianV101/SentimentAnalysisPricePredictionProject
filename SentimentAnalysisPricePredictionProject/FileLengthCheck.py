import csv
from utils.customLogger import setup_logger
from pathlib import Path


logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# this program was created to quickly check the number of entries in
# a data file as at the beginning of the project some files were too large
# to load and measure directly

file_path = 'Data/Bitcoin_tweets.csv'

# Open the file in read mode
with open(file_path, 'r', encoding='utf-8') as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Count the number of rows
    num_rows = sum(1 for row in reader)

logger.info(f"Number of rows in {file_path}: {num_rows}")