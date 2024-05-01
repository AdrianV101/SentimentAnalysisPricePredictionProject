from utils.customLogger import setup_logger
from pathlib import Path
import pandas as pd
import ast
import matplotlib.pyplot as plt

# this script was used to remove any columns that contained only 0's
logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

uncleandf=pd.read_csv("Data/InputData.csv")
logger.info(uncleandf["previous4hrohlcv"])
delete=True
for index, row in uncleandf.iterrows():
    if row["previous4hrohlcv"]==0:
        continue
    else:
        logger.info("At least one entry has non-zero value")
        delete=False
if delete:
    cleandf=uncleandf.drop("previous4hrohlcv", axis=1)
    logger.info("Deleted column")

cleandf.to_csv("../Data/InputData.csv", header=True, index=False)