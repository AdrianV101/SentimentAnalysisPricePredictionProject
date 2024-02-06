import json

from utils.customLogger import setup_logger
from pathlib import Path
import requests
# this program is not in use, nor updated as it was used to access coinAPI for data but
# an unofficial Yahoo Finance API was chosen for use instead
# this is loading API key from .env file to avoid publishing my API key to gitHub (security)
with open('.env', 'r') as f:
    for line in f:
        key, value = line.strip().split('=', 1)
        api_key = value

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger
# generic API URL, can use query parameters period_id. time_start,
# https://rest.coinapi.io/v1/ohlcv/:symbol_id/history?


def fetch_ohlcv(symbol_id, period, time_start, time_end):
    url = (f"https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history?period_id={period}"
           f"&time_start={time_start}&time_end={time_end}")
    headers = {f"X-CoinAPI-Key": {api_key}}

    response = requests.get(url, headers=headers)

    # Check if the response is successful
    if response.status_code == 200:
        if response.content:
            return response.json()
        else:
            print("Response is empty.")
            return None
    else:
        # Handle other HTTP status codes
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None


def fetch_binance_symbols():
    url = "https://rest.coinapi.io/v1/symbols/BINANCE"
    headers = {f"X-CoinAPI-Key": {api_key}}

    response = requests.get(url, headers=headers)

    # Check if the response is successful
    if response.status_code == 200:
        if response.content:
            json_file = open(r"..\Data\coinapi_binance_symbols.json", "w")
            json.dump(response.json(), fp=json_file, indent=2)
            print("response.json")
        else:
            print("Response is empty.")
            return None
    else:
        # Handle other HTTP status codes
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
