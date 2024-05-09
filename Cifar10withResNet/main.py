import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as d
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration, mps is slower for some reason
device = torch.device("cpu")

# Hyper-parameters
num_epochs = 200
batch_size = 32
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = d.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

test_loader = d.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)


def create_resnet18():
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


model = create_resnet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


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

            for i in range(len(labels)):
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

print("Start Training")
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

        if (i + 1) % 2500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f'\nTesting after epoch {epoch + 1}:')
        print("Test Accuracy")
        epoch_acc = test_network(model, test_loader, device)
        print("Training Accuracy")
        train_acc = test_network(model, train_loader, device)
        test_accuracies.append(epoch_acc)
        train_accuracies.append(train_acc)
        print('\n')

print('Finished Training')


print('Final Testing')
final_acc = test_network(model, test_loader, device)
test_accuracies.append(final_acc)
final_train = test_network(model, train_loader, device)
train_accuracies.append(final_train)


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




