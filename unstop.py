from tensorflow.keras.preprocessing.image import load_img, img_to_array   
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2 
import numpy as np                                                                                                                                                    
# Step 1: Load and preprocess the image  
image_path=r"C:\Users\jaanu\Downloads\road.jpg"      
# Load the image and resize it to 224x224  
image = load_img(image_path, target_size=(224, 224)) 
# Convert the image to a NumPy array 
image_array = img_to_array(image) 
# Preprocess the image 
image_array=preprocess_input(image_array) 
# Add batch dimension 
input_image=np.expand_dims(image_array,axis=0) 
# Step 2: Load a pre-trained model  
model = MobileNetV2(weights='imagenet') 
# Step 3: Pass the image through the model the to get predictions 
predictions = model.predict(input_image) 
# Step 4: Decode predictions 
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions  
decoded_predictions = decode_predictions(predictions,top=3)      
# Get top 3 predictions 
# Display the predictions 
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):                                    
    print(f"{i + 1}: {label} (score: {score:.4f})") 
#MACHINE LEARNING LAYERS: 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
def build_cnn_model():                                                                                                                           
    model = Sequential() 
    # Convolutional layer 1                                                                                        
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1))) 
    model.add(MaxPooling2D(pool_size=(2, 2)))                                            
    # Convolutional layer 
    model.add(Conv2D(64, (3, 3), activation='relu'))                                                                                                                                                     
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    # Convolutional layer 
    model.add(Conv2D(128, (3, 3), activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    # Flatten and Fully Connected layers                                                                             
    model.add(Flatten())                                                                                                       
    model.add(Dense(128, activation="relu"))
    # Output layer (binary classification: lane or no lane)        
    model.add(Dense(1, activation='sigmoid')) 
    # Compile the model 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    return model 

# Build the model 
cnn_model = build_cnn_model() 
cnn_model.summary() 
#BINARY CODE THAT FINDS THE LANE: 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
# Load the image using PIL 
image_path = r"C:\Users\jaanu\Downloads\road.jpg"  # Replace with your actual image path 
image = Image.open(image_path).convert("RGBA")  # Ensure the image has 4 channels (RGBA) 
# Convert the image to a NumPy array 
image_np = np.array(image) 
# Let's assume we're working on the left half of the image 
left_half_image = image_np[:, :image_np.shape[1] // 2, :] 
# Create a new channel (e.g., grayscale intensity) and append it to make 4 channels 
# Convert the image to grayscale for the new channel 
gray_channel = np.dot(left_half_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32) 
# Stack the RGBA + Grayscale channels 
left_half_image_4ch = np.dstack((left_half_image, gray_channel))                                                                                                                                                     
# Thresholding: Detect bright pixels that may correspond to white lines in the grayscale channel 
threshold = 200  # Values above this are considered "white" lines 
binary_mask = np.where(gray_channel > threshold, 1, 0) 
# Plot the left half of the original image and the binary mask for detected lines 
plt.subplot(1, 2, 1) 
plt.imshow(left_half_image[:, :, :3])  # Display RGB channels 
plt.title("Left Side Image (RGB)") 
plt.subplot(1, 2, 2) 
plt.imshow(binary_mask, cmap='gray') 
plt.title("Detected Lines (Binary Mask)") 
plt.show() 
# Example CNN Layer 
# Set in_channels=4 to match the RGBA + grayscale input of the image 
conv_layer = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=4, stride=1, padding=1) 
# Convert the left half of the image to a PyTorch tensor 
# Permute to match the expected (batch_size, channels, height, width) format 
left_half_tensor = torch.tensor(left_half_image_4ch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 
# Pass the tensor through the convolutional layer 
conv_output = conv_layer(left_half_tensor) 
# Print details about the convolutional layer and its output 
print("Convolutional Layer Details:") 
print(f"Kernel Size: {conv_layer.kernel_size}") 
print(f"Stride: {conv_layer.stride}") 
print(f"Padding: {conv_layer.padding}") 
print("Output Shape after Convolution:", conv_output.shape) 
# Convert the output to numpy and visualize it 
output_np = conv_output.squeeze().detach().numpy() 
plt.imshow(output_np, cmap='gray') 
plt.title("Convolutional Layer Output") 
plt.show()                                                                           