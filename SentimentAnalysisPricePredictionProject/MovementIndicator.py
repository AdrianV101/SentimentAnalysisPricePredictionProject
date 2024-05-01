from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd

# this script was used to create the price movement that occurred as a separate input feature to others
# there was also another version which separated the price movement into the magnitude and sign as seperate features
logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

file_path= "Data/InputData.csv"

original=pd.read_csv(file_path)
original["movement"]=0

for index, row in original.iterrows():
    original.loc[index, "movement"] = row["current4hrc"]-row["current4hro"]
original.to_csv("../Data/InputData.csv",header=True,index=False)
