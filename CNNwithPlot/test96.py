import torch
import os
from PIL import Image
from torchvision.transforms import Resize, Normalize, ToTensor, Compose, Grayscale, Lambda
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

if not os.path.exists('preprocessed'):
    os.makedirs('preprocessed')

transform = Compose([
    Grayscale(),  # Convert to grayscale
    Resize((28, 28)),  # Resize to 28x28
    ToTensor(),  # Convert to tensor
    # Lambda(lambda x: 1.0 - x),  # Invert the image colors
    Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

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
    
model = ConvNet()

state_dict = torch.load('engr96rotation.pth')
model.load_state_dict(state_dict)
model.eval()


for i in range(0, 10):
    image_path = f'./BlackBackground/{i}.png'
    image = Image.open(image_path)
    image_tensor = transform(image)
    
    # Save the preprocessed image for inspection
    save_image(image_tensor, f'preprocessed/{i}_preprocessed.png')
    
    
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    print(f"Size of the image tensor: {image_tensor.size()}")
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted digit: {predicted_class}")


for i in range(0, 10):
    image_path = f'./BlackBackground/{i}{i}.png'
    image = Image.open(image_path)
    image_tensor = transform(image)
    
    # Save the preprocessed image for inspection
    save_image(image_tensor, f'preprocessed/{i}_preprocessed.png')
    
    
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    print(f"Size of the image tensor: {image_tensor.size()}")
    
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f"Predicted Rotated digit: {predicted_class}")

