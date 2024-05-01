import pandas as pd
from utils.customLogger import setup_logger
from pathlib import Path
logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mplfinance as mpf

# this program trains the Random Forest Regressor model from sci kit learn, and then uses it to predict values


# Load the CSV file into a DataFrame
x_data = pd.read_csv('Data/InputData.csv')
y_data = pd.read_csv('Data/OutputData.csv')

# removing data that isnt in the same range as our training data, as the model struggles to predict these values

x_data=x_data.iloc[:int(len(x_data) * 0.65)]
y_data=y_data.iloc[:int(len(y_data) * 0.65)]
# Separate input features and outputs
X = x_data.drop(columns=['start_time', 'end_time',"current4hrv","current15mv","previous4hrv"])  # Input features
y = y_data.drop(columns=['start_time', 'end_time'])  # outputs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)


# Predict on the testing set
y_pred = rf_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


y_train_times, y_test_times = train_test_split(y_data["start_time"], test_size=0.2, shuffle=False)
plt.scatter(y_train_times, y_train["h"],label="Training values")
plt.scatter(y_test_times,y_test["h"],label="Test Values")
plt.scatter(y_test_times,[row[1] for row in y_pred], label="Predicted Values")
plt.legend()
plt.show()


# plotting candlestick graphs of the real data and predicted data

y_test_dict={"Date":pd.to_datetime(y_test_times),
             "Open":y_test["o"],
             "High":y_test["h"],
             "Low":y_test["l"],
             "Close":y_test["c"]}
y_pred_dict={"Date":pd.to_datetime(y_test_times),
             "Open":[row[0] for row in y_pred],
             "High":[row[1] for row in y_pred],
             "Low":[row[2] for row in y_pred],
             "Close":[row[3] for row in y_pred]}
y_test_plot=pd.DataFrame(y_test_dict)
y_pred_plot=pd.DataFrame(y_pred_dict)
y_test_plot.set_index('Date', inplace=True)
y_pred_plot.set_index('Date', inplace=True)
pd.to_datetime(y_test_plot.index)
pd.to_datetime(y_pred_plot.index)

# Create a new figure
fig = plt.figure()


# Plot the main candlestick chart
ax1 = fig.add_subplot(1, 1, 1)
mpf.plot(y_test_plot, type='candle', style='charles', ax=ax1)

# Plot the additional candlestick chart overlaid on the same axes
mpf.plot(y_pred_plot, type='candle', style='binance', ax=ax1)

# Set the title
ax1.set_title('Candlestick Chart')

# Show the plot
plt.show()
correctup=0
correctdown=0
incorrectup=0
incorrectdown=0
for index, row in y_test_plot.iterrows():
    if row["Close"]-row["Open"] >=0 and y_pred_plot.loc[index,"Close"]-y_pred_plot.loc[index,"Open"]>=0:
        correctup+=1
    if row["Close"]-row["Open"] <0 and y_pred_plot.loc[index,"Close"]-y_pred_plot.loc[index,"Open"]<0:
        correctdown+=1
    if row["Close"]-row["Open"] >=0 and y_pred_plot.loc[index,"Close"]-y_pred_plot.loc[index,"Open"]<0:
        incorrectup+=1
    if row["Close"]-row["Open"] <0 and y_pred_plot.loc[index,"Close"]-y_pred_plot.loc[index,"Open"]>=0:
        incorrectdown+=1
logger.info(f"The model predicted the movement correctly {correctup+correctdown} times out of {correctup+correctdown+incorrectdown+incorrectup} total predictions")
logger.info(f"It predicted the movement up {correctup} times correctly, and down {correctdown} times correctly")
logger.info(f"It predicted a down movement when the market moved up {incorrectup} times, and up movement when market moved down {incorrectdown} times")