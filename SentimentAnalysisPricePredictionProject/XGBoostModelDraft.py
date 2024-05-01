import pandas as pd
from utils.customLogger import setup_logger
from pathlib import Path
logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# this program implements an XGBoost regressor, however it was harder to tune and had the same pitfalls as the Random
# Forest Regression, and so was quickly abandoned.



x_data = pd.read_csv('Data/InputData.csv')
y_data = pd.read_csv('Data/OutputData.csv')


# Separate input features and outputs
X = x_data.drop(columns=['start_time', 'end_time'])  # Input features
y = y_data.drop(columns=['start_time', 'end_time'])  # outputs

#y=y_data["c"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



# Create GBM model
gbm_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10,
                         subsample=0.5, colsample_bytree=0.2, reg_alpha=0.5, reg_lambda=0.5,
                         gamma=0, objective='reg:squarederror', eval_metric='rmse')
#could try grid method on XGB to find best params

# Train the GBM model
gbm_model.fit(X_train, y_train)

# Predict
y_pred = gbm_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
logger.info("Mean Squared Error:", int(mse))


y_train_times, y_test_times = train_test_split(y_data["start_time"], test_size=0.2, shuffle=False)
plt.scatter(y_test_times,y_test["h"],label="Test Values")
plt.scatter(y_test_times,[row[1] for row in y_pred], label="Predicted Values")
plt.legend()
plt.show()