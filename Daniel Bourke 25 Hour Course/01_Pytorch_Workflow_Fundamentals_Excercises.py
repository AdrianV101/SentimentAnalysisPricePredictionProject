import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from colorlog import ColoredFormatter
import datetime
from pathlib import Path

#setting up logging
program_name='00_fundamentals'
# Create a logger for your module
logger = logging.getLogger(program_name)
logger.setLevel(logging.DEBUG)

# Create a file handler with dynamic file name
current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file_name = f'Logs/{program_name}_{current_datetime}.log'
file_handler = logging.FileHandler(log_file_name)

# Set the log message format
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

#get colors in console
log_colors = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}
#add console handler
console_handler = logging.StreamHandler()
console_formatter=ColoredFormatter("%(log_color)s%(asctime)s [%(levelname)s] - %(message)s",datefmt='%Y-%m-%d %H:%M:%S',log_colors=log_colors)
console_handler.setFormatter(console_formatter)

#add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

#making program run on GPU not CPU, if possible
if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_device(device)
else:
    device = "cpu"
    torch.set_default_device(device)

#========================================START OF EXCERCISES==========================================================
torch.manual_seed(100)
#making data
weight= 0.3
bias=0.9
x_set=torch.arange(0,100,1)
y_set=x_set*weight+bias
x_set=torch.squeeze(x_set)
y_set=torch.squeeze(y_set)

#splitting data
training_split=int(0.8*len(x_set))
x_train=x_set[:training_split]
y_train=y_set[:training_split]
x_test=x_set[training_split:]
y_test=y_set[training_split:]

plt.scatter(x_train.cpu(),y_train.cpu(),s=3)
plt.scatter(x_test.cpu(), y_test.cpu(),s=3)
logger.info(x_test.device)
plt.show()



#TASK 2

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)


model_1=LinearModel()
logger.info(model_1.state_dict())

loss_fn=nn.L1Loss()

optimizer=torch.optim.SGD(params=model_1.parameters(),lr=0.00001)
epochs=100000
for epoch in range(epochs):
    model_1.train()
    y_pred=model_1(x_train)
    loss=loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2000 ==0:
        model_1.eval()
        test_pred=model_1(x_test)
        test_loss=loss_fn(test_pred,y_test.type(torch.float))
        logger.info(f"The loss at epoch {epoch} is {test_loss}")

#make final predictions
model_1.eval()
with torch.inference_mode():
    predictions=model_1(x_test)
predictions=predictions.cpu()
predictions.detach().numpy()

torch.set_default_device("cpu") # need to change to CPU to work with matplotlib and numpy etc.
plt.scatter(x_train.cpu(),y_train.cpu(),s=3)
plt.scatter(x_test.cpu(),y_test.cpu(),s=3)
plt.scatter(x_test.cpu(),predictions.cpu(),s=3)
plt.show()

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_excercises_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
logger.info(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)



