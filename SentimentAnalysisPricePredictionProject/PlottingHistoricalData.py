import json
from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

#this program simply plots the historical data on bitcoin
file_path= "Data/dataset.json"
with open(file_path, 'r') as file:
    data=json.load(file)
    data=data["4h"]
data["ot"]=pd.to_datetime(data["ot"], unit="ms")
data["ct"]=pd.to_datetime(data["ct"], unit="ms")

df=pd.DataFrame(data)

plt.plot(df["ot"],df["o"])
plt.show()
