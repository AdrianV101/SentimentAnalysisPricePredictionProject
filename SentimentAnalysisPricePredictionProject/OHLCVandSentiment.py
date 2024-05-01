from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import json
import ast
from matplotlib import pyplot as plt
import time

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# this program was made in order to create the input data for the predictive models, finding values such as OHLCVs and
# averages
file_path= "Data/dataset.json"
with open(file_path, 'r') as file:
    data=json.load(file)
    data=data["5m"]
data["ot"]=pd.to_datetime(data["ot"], unit="ms")
data["ct"]=pd.to_datetime(data["ct"], unit="ms")

historical=pd.DataFrame(data)

sentiments=pd.read_csv("Data/CompressedSentimentForInput.csv", parse_dates=["start_time", "end_time"])
sentiments["current4hrohlcv"] = 0
sentiments["current15mohlcv"] = 0
sentiments["previous4hrohlcv"] = 0
sentiments["average8hr"] = 0
sentiments["average24hr"] = 0
s_length=len(sentiments)
for s_index, s_row in sentiments.iterrows():
    logger.info(f"{s_index} out of {s_length}")
    volume=0
    # this ensures that we always round down the start time (and thus end time), so that we don't use information from future
    start_time = s_row["start_time"]-pd.Timedelta(minutes=2,seconds=30)
    start_time =start_time.round(freq="5min")
    end_time = start_time + pd.Timedelta(hours=4)
    high_value=0
    low_value=1000000000000000000000

    # code for OHLCV of last 4 hours
    start = historical.query('ot == @start_time')
    end = historical.query("ot == @end_time")
    window = historical.iloc[start.index.values[0]:end.index.values[0]]
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
    current4hrohlcv={"o":open_value,"h":high_value,"l":low_value,"c":close_value,"v":volume}

    # code for finding the mean price of last 8 hours
    start_time -= pd.Timedelta(hours=4)
    start = historical.query('ot == @start_time')
    window = historical.iloc[start.index.values[0]:end.index.values[0]]
    entries = 0
    for index, row in window.iterrows():
        entries += 1
        if row["ct"]<start_time or row["ot"]>end_time:
            logger.warning("Somehow outside of desire window")
            time.sleep(1)
            continue
        if row["ot"] == start_time:
            total = row["o"] + row["c"]
            entries+=1
        else:
            total += row["c"]
    average8hr = total/entries

    # code for finding the mean price of last 24 hours
    start_time = end_time - pd.Timedelta(hours=24)
    start = historical.query('ot == @start_time')
    window = historical.iloc[start.index.values[0]:end.index.values[0]]
    entries = 0
    total=0
    for index, row in window.iterrows():
        entries += 1
        if row["ct"] < start_time or row["ot"] > end_time:
            logger.warning("Somehow outside of desire window")
            time.sleep(1)
            continue
        if row["ot"] == start_time:
            total = row["o"] + row["c"]
            entries += 1
        else:
            total += row["c"]
    average24hr = total / entries
    #code for previous 15 minute OHLCV
    search=end_time-pd.Timedelta(minutes=15)
    fullcurrent15mohlcv = historical.query("ot == @search")
    current15mohlcv = {"o":fullcurrent15mohlcv["o"].values[0],"h":fullcurrent15mohlcv["h"].values[0],"l":fullcurrent15mohlcv["l"].values[0],"c":fullcurrent15mohlcv["c"].values[0],"v":fullcurrent15mohlcv["v"].values[0]}

    # code for previous (as in 4-8 hours ago, not 0-4 hours) 4hr OHLCV
    high_value=0
    low_value=1000000000000000000000
    start_time=end_time-pd.Timedelta(hours=8)
    end_time-=pd.Timedelta(hours=4)
    start = historical.query('ot == @start_time')
    end = historical.query("ot == @end_time")
    window = historical.iloc[start.index.values[0]:end.index.values[0]]
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
    previous4hrohlcv={"o":open_value,"h":high_value,"l":low_value,"c":close_value,"v":volume}

    sentiments.loc[s_index, 'average8hr'] = average8hr
    sentiments.loc[s_index, 'average24hr'] = average24hr
    sentiments.loc[s_index, "current15mo"] = current15mohlcv["o"]
    sentiments.loc[s_index, "current15mh"] = current15mohlcv["h"]
    sentiments.loc[s_index, "current15ml"] = current15mohlcv["l"]
    sentiments.loc[s_index, "current15mc"] = current15mohlcv["c"]
    sentiments.loc[s_index, "current15mv"] = current15mohlcv["v"]
    sentiments.loc[s_index, "current4hro"] = current4hrohlcv["o"]
    sentiments.loc[s_index, "current4hrh"] = current4hrohlcv["h"]
    sentiments.loc[s_index, "current4hrl"] = current4hrohlcv["l"]
    sentiments.loc[s_index, "current4hrc"] = current4hrohlcv["c"]
    sentiments.loc[s_index, "current4hrv"] = current4hrohlcv["v"]
    sentiments.loc[s_index, "previous4hro"] = previous4hrohlcv["o"]
    sentiments.loc[s_index, "previous4hrh"] = previous4hrohlcv["h"]
    sentiments.loc[s_index, "previous4hrl"] = previous4hrohlcv["l"]
    sentiments.loc[s_index, "previous4hrc"] = previous4hrohlcv["c"]
    sentiments.loc[s_index, "previous4hrv"] = previous4hrohlcv["v"]


sentiments.to_csv("../Data/InputData.csv", header=True, index=False)



