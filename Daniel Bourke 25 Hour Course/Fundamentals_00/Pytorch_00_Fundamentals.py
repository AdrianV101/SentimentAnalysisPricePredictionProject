#=====================================IMPORTS AND LOGGING CONFIG======================================================#
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from colorlog import ColoredFormatter
import datetime

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
formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

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

#=====================================TENSOR BASICS======================================================#
#making program run on GPU not CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
#scalar
scalar= torch.tensor(7)
logger.info(scalar.ndim)
logger.info(scalar.item())

#vector
vector = torch.tensor([7,7])
logger.info(vector)
logger.info(vector.ndim)

#MATRIX
MATRIX = torch.tensor([[7,8],[9,10]])

logger.info(MATRIX)
logger.info(MATRIX.ndim)

#TENSOR, 1 dimension of 3 by 3 matrix
TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])

logger.info(TENSOR)
logger.info(TENSOR.ndim)

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
logger.info(random_tensor)
logger.info(random_tensor.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
logger.info(random_image_size_tensor.shape)
logger.info(random_image_size_tensor.ndim)

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
logger.info(zeros)
logger.info(zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
logger.info(ones)
logger.info(ones.dtype)

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
logger.info(zero_to_ten)

# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
logger.info(ten_zeros)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded
logger.info(float_32_tensor.shape)
logger.info(float_32_tensor.dtype)
logger.info(float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16)
logger.info(float_16_tensor.dtype)
#=====================================TENSOR OPERATIONS======================================================#
# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
logger.info(f'Addition: {tensor+10}') #addition
logger.info(f'Normal multiplication: {tensor*10}') #normal multiplication
logger.info(f'Subtraction: {tensor-10}') #subtraction
logger.info(f'Matrix multiplication {torch.mm(torch.tensor([[2,3],[4,5]]),torch.tensor([[6,7],[8,9]]))}')#matrix multiplication

#=====================================TENSOR OPERATIONS======================================================#
# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input
                         out_features=6) # out_features = describes outer value
x = torch.tensor([[1,2],
                  [3,4],
                  [5,6]],dtype=torch.float32)
output = linear(x)
logger.info(f"Input shape: {x.shape}\n")
logger.info(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# Create a tensor
x = torch.arange(0, 100, 10,dtype=(torch.float32))

logger.info(f"Minimum: {x.min()}")
logger.info(f"Maximum: {x.max()}")
logger.info(f"Mean: {x.mean()}") # won't work without float datatype
logger.info(f"Sum: {x.sum()}")
logger.info(f"Index where max value occurs: {x.argmax()}")
logger.info(f"Index where min value occurs: {x.argmin()}")

#=========================================================EXCERCISES=======================================================
tensor=torch.rand([7,7])
logger.info("Excercises")
logger.info(tensor.shape)

tensor=torch.rand([1,1,1,10])
tensor=torch.squeeze(tensor)
logger.info(tensor.shape)


