from pymongo import MongoClient
from PIL import Image
import os

# Connect to MongoDB locally
client = MongoClient("mongodb://localhost:27017/TrafficData")

# Access the TrafficData database and Images collection
db = client["TrafficData"]
images_collection = db["Images"]

# Function to convert image to binary and upload to MongoDB
def save_image_to_mongo(file_path):
    with open(file_path, "rb") as image_file:  # Open image file in binary mode
        image_binary = image_file.read()  # Read image as binary data
        image_data = {  # Create a document with image details
            "filename": os.path.basename(file_path),  # File name
            "image": image_binary  # Image binary data
        }
        images_collection.insert_one(image_data)  # Insert document into MongoDB

# Specify the folder with images
dataset_folder = "C:/Users/harsh/OneDrive/Desktop/FALL-2024/603/project/dataset/CrashBest"  
# Loop through the images and upload each one to MongoDB
for file_name in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, file_name)
    save_image_to_mongo(file_path)

print("Images have been uploaded to MongoDB.")
