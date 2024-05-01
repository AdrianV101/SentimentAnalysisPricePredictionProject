import json

from utils.customLogger import setup_logger
from pathlib import Path

logger = setup_logger(Path(__file__).name[:-3])
# this program is not in use, nor updated as it was used to access coinAPI for data but
# an unofficial Yahoo Finance API was chosen for use instead
symbol_list_file = open(r"../SentimentAnalysisPricePredictionProject/Data\coinapi_binance_symbols.json", "r")   # load json file of all coinapi binance symbols
symbol_list = json.load(symbol_list_file)
symbol_list_file.close()

usdt_list = []
symbol_dictionary = {}
for symbol in symbol_list:   # filter through these symbols and isolate those which are USDT
    # SPOT and have a high volume (for liquidity)
    try:
        if (("USDT" in symbol['symbol_id'])
                and (symbol['volume_1hrs_usd'] > 100000)
                and (symbol["symbol_type"] == "SPOT")):
            usdt_list.append({
                "symbol_id": symbol["symbol_id"],
                "data_start": symbol["data_start"],
                "data_end": symbol["data_end"],
                "volume_1hrs_usd": symbol["volume_1hrs_usd"]

            })
    except KeyError:
        logger.warning(f"Key error for {symbol['symbol_id']}")
usdt_list_file = open(r"../SentimentAnalysisPricePredictionProject/Data\binance_usdt_symbols.json", "w")
json.dump(usdt_list, fp=usdt_list_file, indent=2)
usdt_list_file.close()
logger.info(f"Filtered {len(usdt_list)} suitable coins")
