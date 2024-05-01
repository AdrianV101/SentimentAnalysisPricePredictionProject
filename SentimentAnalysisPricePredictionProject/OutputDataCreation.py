from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import json
import ast
from matplotlib import pyplot as plt
import time

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger
# this program was made in order to create the output data to train the models on, which is the OHLC of the next period

file_path= "Data/dataset.json"
with open(file_path, 'r') as file:
    data=json.load(file)
    data=data["5m"]
data["ot"]=pd.to_datetime(data["ot"], unit="ms")
data["ct"]=pd.to_datetime(data["ct"], unit="ms")

historical=pd.DataFrame(data)

sentiments=pd.read_csv("Data/CompressedSentimentForInput.csv", parse_dates=["start_time", "end_time"])
outputdf=pd.DataFrame()
outputdf["start_time"]=0
outputdf["end_time"]=0
outputdf["o"]=0
outputdf["h"]=0
outputdf["l"]=0
outputdf["c"]=0


s_length=len(sentiments)
for s_index, s_row in sentiments.iterrows():
    logger.info(f"{s_index} out of {s_length}")
    volume=0
    # this ensures that we always round down the start time (and thus end time), so that we don't use information from future
    start_time = s_row["end_time"]+pd.Timedelta(minutes=2,seconds=30)
    start_time =start_time.round(freq="5min")
    end_time = start_time + pd.Timedelta(hours=4)
    high_value=0
    low_value=1000000000000000000000
    # code for OHLCV of last 4 hours
    start = historical.query('ot == @start_time')
    end = historical.query("ot == @end_time")
    try:
        window = historical.iloc[start.index.values[0]:end.index.values[0]]
    except:
        logger.warning(f"Error with start time {start_time}, skipping")
        continue
    for index, row in window.iterrows():
        if row["ct"]<start_time or row["ot"]>end_time:
            logger.warning("Somehow outside of desire window")
            continue
        volume += row["v"]
        if row["h"]>high_value:
            high_value = row["h"]
        if row["l"]<low_value:
            low_value = row["l"]
        if row["ot"] == start_time:
            open_value = row["o"]
        if row["ct"].round(freq="5min") == end_time:
            close_value = row["c"]

    new_row={"start_time":start_time,"end_time":end_time,"o":open_value,"h":high_value,"l":low_value,"c":close_value}

    outputdf = outputdf._append(new_row, ignore_index=True)


outputdf.to_csv("../Data/OutputData.csv", header=True, index=False)