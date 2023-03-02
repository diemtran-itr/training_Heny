import torch
from data_utils import EcgDataset
from model import EcgNet


model = EcgNet(num_classes=5)
model.load_state_dict(torch.load("mit"))
model.eval()

test_dataset = EcgDataset("data/test/", "test.csv")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print("Test set accuracy: {:.2f}%".format(accuracy * 100))
