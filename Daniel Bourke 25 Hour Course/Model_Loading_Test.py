import torch
import matplotlib.pyplot as plt
from utils.customLogger import setup_logger
from pathlib import Path
from utils.ModelClasses import LinearModel

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# configuring CPU/GPU usage
if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_device(device)
else:
    device = "cpu"
    torch.set_default_device(device)


# ========================================START OF EXCERCISES==========================================================
torch.manual_seed(100)
# making data
weight = 0.3
bias = 0.9
x_set = torch.arange(0, 100, 1)
y_set = x_set*weight+bias
x_set = torch.squeeze(x_set)
y_set = torch.squeeze(y_set)

# splitting data
training_split = int(0.8*len(x_set))
x_train = x_set[:training_split]
y_train = y_set[:training_split]
x_test = x_set[training_split:]

plt.scatter(x_train.cpu(), y_train.cpu(), s=3)
plt.show()

# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model_1 = LinearModel()

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_excercises_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Load model state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")

loaded_model_1.eval()
with torch.inference_mode():
    predictions = loaded_model_1(x_test)
predictions = predictions.cpu()
predictions.detach().numpy()

plt.scatter(x_train.cpu(), y_train.cpu(), s=3)
plt.scatter(x_test.cpu(), predictions.cpu(), s=3)
plt.show()
