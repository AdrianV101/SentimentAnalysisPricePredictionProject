import json
import yfinance as yf
from utils.customLogger import setup_logger
from pathlib import Path
import boto3
import time

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

keys = []
# accessing AWS keys from .env file, which is to avoid publishing them on Git (security)
with open(r'..\.env', 'r') as f:
    for line in f:
        key, value = line.strip().split('=', 1)
        keys.append(value)

client = boto3.client('dynamodb', aws_access_key_id=keys[0], aws_secret_access_key=keys[1], region_name="eu-west-2")

json_file = open(r"../SentimentAnalysisPricePredictionProject/Data\crypto_list.json", "r")  # load crypto list
cryptocurrencies = json.load(json_file)
ticker_list = []
for i in range(len(cryptocurrencies)):
    ticker = cryptocurrencies[i]+"-USD"
    ticker_list.append(ticker)
logger.info(ticker_list)
'''
print(btc.info)
history = btc.history(period="max", interval="1d")
print(history)
print(type(history.index[0]))'''
for j in range(len(ticker_list)):
    start_time = time.time()
    ticker = yf.Ticker(ticker_list[j])
    logger.info(ticker.info)
    if "cryptocurrency" not in ticker.info["description"]:
        logger.warning(f"Warning {ticker.info['name']} may not be cryptocurrency")
    history = ticker.history(period="max", interval="1d")
    for i in range(len(history)):
        item = {
            'symbol': {'S': ticker.info['name']},
            'timestamp': {'S': str(history.index[i])},
            'Open': {'S': str(history.iloc[i]["Open"])},
            'High': {'S': str(history.iloc[i]["High"])},
            'Low': {'S': str(history.iloc[i]["Low"])},
            'Close': {'S': str(history.iloc[i]["Close"])},
            'Volume': {'S': str(history.iloc[i]["Volume"])},
        }

        # PutItem operation to insert data into DynamoDB table
        response = client.put_item(
            TableName="crypto_historical_data",
            Item=item
        )

        # Check response
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            logger.warning(f"Response {i} for {ticker_list[j]}not successful: {response}")
    end_time = time.time()
    logger.info(f"Ticker: {ticker.info['name']}. Took {end_time-start_time} seconds to upload {len(history)} entries")
