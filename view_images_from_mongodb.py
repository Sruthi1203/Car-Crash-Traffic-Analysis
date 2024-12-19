from pymongo import MongoClient
from PIL import Image
import base64
import io

# MongoDB connection
connection_string = "mongodb://localhost:27017/TrafficData"
client = MongoClient(connection_string)
db = client["TrafficData"]
collection = db["Images"]

def display_image_from_mongodb(image_doc):
    """
    Decodes and displays an image from MongoDB document
    """
    try:
        # Get filename from document
        filename = image_doc.get('filename', 'unknown')
        print(f"\nProcessing image: {filename}")
        
        # Get image data 
        image_binary = image_doc.get('image')
        
        if image_binary is None:
            print("No image data found")
            return

        # Create and display image
        image = Image.open(io.BytesIO(image_binary))
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        image.show()

    except Exception as e:
        print(f"Error displaying image: {str(e)}")

# Print collection information
print(f"Total documents in collection: {collection.count_documents({})}")

# Process each image
for image_doc in collection.find({}):
    display_image_from_mongodb(image_doc)

# Close connection
client.close()
