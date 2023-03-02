import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils import load_data, preprocess_data
from model import EcgNet


torch.manual_seed(0)

data_dir = "mit"
records, labels = load_data(data_dir)

records = preprocess_data(records)

train_records, test_records, train_labels, test_labels = train_test_split(records, labels, test_size=0.2, random_state=0)

train_set = [(record, label) for record, label in zip(train_records, train_labels)]
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = [(record, label) for record, label in zip(test_records, test_labels)]
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = EcgNet(num_classes=5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    train_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print("Epoch {} - Training loss: {:.4f}".format(epoch + 1, train_loss))

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
