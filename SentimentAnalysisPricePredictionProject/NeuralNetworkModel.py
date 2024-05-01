import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.customLogger import setup_logger
from pathlib import Path
import matplotlib.pyplot as plt
import mplfinance as mpf

# this program was used to develop the neural network, the one below is the one with the best performance after many
# rounds of trial and error

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available")
    # Additional GPU-related information
    print("GPU Name:", tf.test.gpu_device_name())
    print("GPU Memory:", tf.test.gpu_device_name())
else:
    print("GPU is not available")

x_data = pd.read_csv('Data/InputData.csv')
y_data = pd.read_csv("Data/OutputData.csv")
#x_data=x_data.iloc[:int(len(x_data) * 0.65)]   # Uncomment if you want to compare the models performance when
#y_data=y_data.iloc[:int(len(y_data) * 0.65)]   # predicting values within the previously seen range vs outside range
# Separate features and target variable
X = x_data.drop(columns=['start_time',"end_time","movement"])
y = y_data.drop(columns=['start_time',"end_time"])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)



initial_learning_rate = 0.01  # Initial learning rate
decay_steps = 100            # Number of steps before each learning rate decay
decay_rate = 0.1            # Decay rate

# Define the learning rate schedule
def lr(epoch):
    if epoch<=50:
        return 0.01
    elif epoch<=200:
        return 0.001
    elif epoch<=1000:
        return 0.0001
    else:
        return 0.0001

# Create a learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr)


# Define and compile your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(X_train.shape[1], activation='linear', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(24, activation='linear'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(4, activation='linear')])  # Output layer with num_outputs neurons
model.compile(optimizer='adam', loss='mse')

# Train the model with the learning rate scheduler callback
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_train_times, y_test_times = train_test_split(y_data["start_time"], test_size=0.2, shuffle=False)

y_test_dict={"Date":pd.to_datetime(y_test_times),
             "Open":y_test["o"],
             "High":y_test["h"],
             "Low":y_test["l"],
             "Close":y_test["c"]}
y_pred_dict={"Date":pd.to_datetime(y_test_times),
             "Open":[row[0] for row in y_pred], #this is known and so technically doesn't have to be predicted
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

# Assuming y_test_plot and y_pred_plot are your two DataFrames containing candlestick data

# Plot the main candlestick chart
ax1 = fig.add_subplot(1, 1, 1)
mpf.plot(y_test_plot, type='candle', style='charles', ax=ax1)

# Plot the additional candlestick chart overlaid on the same axes
mpf.plot(y_pred_plot, type='candle', style='binance', ax=ax1)

# Set the title
ax1.set_title('Candlestick Chart')

# Show the plot
plt.show()
