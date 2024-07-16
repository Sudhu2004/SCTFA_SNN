import streamlit as st
# imports
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import itertools
from segmentation_models_pytorch.base.modules import SEModule,sSEModule

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50
# Define Network
# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers

        # self.block_1 = nn.Sequential(nn.Conv2d(1, 12, 5),
        #             nn.MaxPool2d(2),
        #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        #             nn.Conv2d(12, 64, 5),
        #             nn.MaxPool2d(2),
        #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        #             nn.Flatten(),
        #             nn.Linear(64*4*4, 10),
        #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True))
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # self.conv2 = nn.Conv2d(12, 64, 5)
        # self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(12*12*12, 10)
        self.relu_ = nn.ReLU()
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)


    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.relu_(self.conv1(x)), 2)
        sSE = sSEModule(cur1.shape[1])
        sSE = sSE.to(device)
        sSE_out = sSE.forward(cur1)

        SE = SEModule(cur1.shape[1],reduction = 4)
        SE = SE.to(device)
        SE_out = SE.forward(cur1)
        # cur2 = F.max_pool2d(self.conv2(spk1), 2)
        # spk2, mem2 = self.lif2(cur2, mem2)

        result = sSE_out * SE_out

        spk1, mem1 = self.lif1(result, mem1)
        batch_size = spk1.size(0)
        cur3 = self.fc1(spk1.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)


        # x = self.block_1(x)
        # spk1, mem1 = self.block_1(x)
        # spk1, mem1 = self.lif1(cur1, mem1)


        return  spk3, mem3
    

class AttentionMapGenerator(nn.Module):
    def __init__(self, input_shape, temperature_coefficient):
        super(AttentionMapGenerator, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(input_shape[0] * input_shape[1], 64)
        self.dense2 = nn.Linear(64, input_shape[0] * input_shape[1])
        self.temperature_coefficient = temperature_coefficient

    def forward(self, inputs):
        flattened_inputs = self.flatten(inputs)
        # print(flattened_inputs.to('cuda'))

        x = F.relu(self.dense1(flattened_inputs))
        x = self.dense2(x)
        x = torch.sigmoid(x / self.temperature_coefficient)
        return x.view(-1, *input_shape)
    

# Example usage
input_shape = (28, 28)  # Example input shape, modify as needed
temperature_coefficient = 0.1  # Example temperature coefficient, modify as needed

attention_map_generator = AttentionMapGenerator(input_shape, temperature_coefficient)
# Example input tensor
input_tensor = torch.randn(1, *input_shape)
# Generate attention map
attention_map = attention_map_generator(input_tensor)
print(attention_map.shape)  # Example output shape: torch.Size([1, 28, 28])

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net
    input_shape = data.shape[2:]
    temperature_coefficient = 0.1

    data = data.to(device)
    attention_map_generator = AttentionMapGenerator(input_shape, temperature_coefficient).to(device)

    for step in range(num_steps):
        # Generate attention map

        attention_map = attention_map_generator(data)
        attention_map = attention_map.to(device)
        # Apply attention to the input data
        data_attended = data * attention_map.unsqueeze(1)

        # Forward pass through the network
        spk_out, mem_out = net(data_attended)

        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)
# Load the model

@st.cache_resource
def load_model():
    model_path = 'snn_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    net = Net().to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_model()
net.eval()


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming `test_loader`, `net`, and `num_steps` are already defined


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)


from PIL import Image
import torchvision.transforms as transforms
import torch

@st.cache_resource
def input_image(image_path):
    # Define the preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image to 28x28
        transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
        transforms.ToTensor(),  # Convert the image to a tensor
    ])

    # Load your custom image
    img_path = image_path
    image = Image.open(img_path)

    image = preprocess(image)


    image = image.unsqueeze(0)  # Add batch dimension

    # Move the image to the same device as the model
    image = image.to(device)

    return image
# Streamlit App
st.title("SNN Image Classification")
import os
import time

def apply_threshold(image_array, threshold=0.5):
    # Apply simple thresholding
    image_array = np.where(image_array > threshold, 1.0, 0.0)
    return image_array


def invert_image(image_array):
    # Invert the image (if necessary)
    image_array = 1.0 - image_array
    return image_array

import cv2

def apply_blur(image_array):
    blurred_image = cv2.GaussianBlur(image_array, (5, 5), 5)
    return blurred_image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 32)  # Adjusted input size after pooling
        self.fc2 = nn.Linear(32, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to preprocess the image
def simple_cnn_preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to load the model
def simple_cnn_load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    model = SimpleCNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Function to preprocess the image
def simple_cnn_preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    print(image.shape)
    return image

# Function to predict image label
@st.cache
def simple_cnn_predict_image(image, model):
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    end_time = time.time()
    return predicted, end_time - start_time

import psutil

def monitor_resources():
    # Monitor CPU and memory usage
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent



uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    with torch.no_grad():
        image = Image.open(uploaded_file).convert('L')

        # Save the image to a temporary file path
        temp_image_path = "number.png"

        # Convert the image to a numpy array
        img_array = np.array(image)
        
        # Normalize the pixel values to be between 0 and 1
        img_array = img_array.astype('float32') / 255.0
        
        processed_image = apply_threshold(img_array)
        processed_image = invert_image(processed_image)
        print(processed_image)
        processed_image = apply_blur(processed_image)

        image.save(temp_image_path)
        net.eval()  # Set the model to evaluation mode
        # image = input_image(f"/content/segmented_images/segment_{i}.png")

        start = time.time()
        image = input_image(uploaded_file)

        spk_rec, _ = forward_pass(net, num_steps, image)
        _, idx = spk_rec.sum(dim=0).max(1)

        end = time.time()
        prediction_time = end - start

        st.write(f"Prediction Time: {prediction_time:.4f} seconds")
        st.write(f"SCTFA-SNN Predicted = {idx.item()}")
        snn_cpu_percent, snn_memory_percent = monitor_resources()
        print(f"CPU Usage: {snn_cpu_percent}%")
        print(f"Memory Usage: {snn_memory_percent}%")
        st.image(processed_image, caption='Processed Image', use_column_width=True)

    

    
    # image = image.detach().cpu().numpy()

    # st.image(processed_image.squeeze(), caption='Processed Image', use_column_width=True)

    model_path = 'simple_cnn.pth'  # Replace with your .pth file path
    model = simple_cnn_load_model(model_path)
    # Generate large random matrices (adjust size as needed)
    # Generate large random matrices (adjust size as needed)


    # Preprocess the image
    image = Image.open(uploaded_file)
    image = input_image(uploaded_file)

    # Get the prediction and inference time
    predicted_class, inference_time = simple_cnn_predict_image(image, model)
    cnn_cpu_percent, cnn_memory_percent = monitor_resources()
    st.write(f"Simple CNN Prediction: {predicted_class.item()}")
    print(f"CPU Usage: {cnn_cpu_percent}%")
    print(f"Memory Usage: {cnn_memory_percent}%")

    
    # # Display results
    # st.write(f"Prediction: {predicted_class.item()}")
    # st.write(f"Inference Time: {inference_time:.4f} seconds")


    # Display attention map
    os.remove(temp_image_path)
    

    
    image_path = f"Confusion Matrix.png" 
    image = Image.open(image_path)

    # Bar plot for CPU Percentages
    st.subheader('CPU Usage Comparison: CNN vs SNN')
    fig_cpu, ax_cpu = plt.subplots(figsize=(8, 6))
    ax_cpu.bar(['CNN', 'SNN'], [cnn_cpu_percent, snn_cpu_percent], color=['blue', 'green'])
    ax_cpu.set_xlabel('Models')
    ax_cpu.set_ylabel('CPU Percent')
    ax_cpu.set_ylim(0, 100)  # Adjust ylim based on your data range
    st.pyplot(fig_cpu)

    st.image(image, caption=f'confusion matrix', use_column_width=True)
