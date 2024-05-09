import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as d
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Normalize, ToTensor, RandomRotation, Compose, CenterCrop, Grayscale
import matplotlib.pyplot as plt
import numpy as np

# Device configuration, mps is slower for some reason
device = torch.device("cpu")

# Hyper-parameters
num_epochs = 22
batch_size = 64
learning_rate = 0.003

# We transform them to Tensors of normalized range [-1, 1]
transform = Compose([
    Resize((28, 28)),  # Resize to a fixed size
    RandomRotation(degrees=(-120, 120)),  # Randomly rotate images in the range (1, 359) degrees
    CenterCrop((28, 28)),  # Center crop to the desired size (e.g., 28x28 for MNIST).
    ToTensor(),  # Convert to tensor
    Normalize((0.5,), (0.5,)),  # Normalize with mean = 0.5, std = 0.5 for MNIST
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = d.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = d.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


print("Data loaded")
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Change input channels to 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Adjusted fully connected layer size because MNIST images are smaller
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjust the input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 16 * 4 * 4)  # Adjust the flatten size
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def test_network(model, test_loader, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(10)]
        n_class_samples = [0 for _ in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
        return acc


losses = []
test_accuracies = []
train_accuracies = []
n_total_steps = len(train_loader)

# Training
print('Training')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 67 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            losses.append(loss.item())

    print(f'\nTesting after epoch {epoch + 1}:')
    # Makes sure that the model is not too biased the training data
    epoch_acc = test_network(model, test_loader, device)
    print('\n')
    train_acc = test_network(model, train_loader, device)
    test_accuracies.append(epoch_acc)
    train_accuracies.append(train_acc)
    print('\n')

print('Finished Training')

torch.save(model.state_dict(), 'engr96rotation.pth')

print('Final Testing')
final_acc = test_network(model, test_loader, device)
test_accuracies.append(final_acc)
final_train = test_network(model, train_loader, device)
train_accuracies.append(final_train)



# with torch.no_grad():
#     output = model(image)
#     predicted_class = torch.argmax(output, dim=1).item()
#     print(f"Predicted digit: {predicted_class}")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Step (2000 iterations)')
plt.ylabel('Loss')
plt.title('Loss Over Time')

# Plot the accuracies over time
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test')
plt.plot(train_accuracies, label='Train')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracies Over Time')
plt.legend()

plt.tight_layout()
plt.show()




