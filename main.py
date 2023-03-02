import numpy as np
import torch
from data_utils import preprocess_data
from model import EcgNet


model = EcgNet(num_classes=5)
model.load_state_dict(torch.load("mit"))
model.eval()

record_path = "mit"
record = np.fromfile(record_path, dtype=np.int16)
record = preprocess_data(record)

with torch.no_grad():
    output = model(torch.Tensor(record).unsqueeze(0))
    _, predicted = torch.max(output.data, 1)

print("Predicted rhythm: {}".format(predicted.item()))
