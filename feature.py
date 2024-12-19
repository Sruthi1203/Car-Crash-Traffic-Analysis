# feature_extraction.py
from pymongo import MongoClient
import cv2
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrafficFeatureExtractor:
    def __init__(self):
        try:
            # Connect to MongoDB
            self.client = MongoClient('mongodb://localhost:27017/')
            self.traffic_data = self.client['TrafficData']
            self.db = self.client['TrafficAnalysis']
            
            # Get collections
            self.images_collection = self.traffic_data['Images']
            self.features_collection = self.db['extracted_features']
            
            # Initialize background subtractor
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
            
            logger.info(f"Connected to MongoDB. Total images: {self.images_collection.count_documents({})}")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def extract_features_from_image(self, image):
        """Extract traffic-related features from an image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(gray)
            
            # Calculate traffic density
            density = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
            
            # Vehicle detection
            _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_vehicle_area = 500
            vehicle_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > min_vehicle_area)
            
            return {
                'vehicle_count': vehicle_count,
                'traffic_density': float(density),
                'processing_timestamp': datetime.now(),
                'num_contours': len(contours)
            }
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None

    def process_images(self, batch_size=10):
        """Process images and store extracted features"""
        try:
            # Find unprocessed images
            cursor = self.images_collection.find({
                "processed": {"$ne": True}
            }).limit(batch_size)
            
            processed_count = 0
            
            for doc in cursor:
                try:
                    logger.info(f"Processing image: {doc['filename']}")
                    
                    # Get image bytes directly
                    img_bytes = doc['image']
                    
                    try:
                        # Convert bytes to numpy array
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is None:
                            logger.error(f"Failed to decode image for {doc['filename']}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Image decoding error for {doc['filename']}: {str(e)}")
                        continue
                    
                    # Extract features
                    features = self.extract_features_from_image(image)
                    
                    if features:
                        # Store features
                        feature_doc = {
                            'image_id': doc['_id'],
                            'filename': doc['filename'],
                            **features
                        }
                        
                        # Insert features
                        self.features_collection.update_one(
                            {'image_id': doc['_id']},
                            {'$set': feature_doc},
                            upsert=True
                        )
                        
                        # Mark as processed
                        self.images_collection.update_one(
                            {'_id': doc['_id']},
                            {'$set': {'processed': True}}
                        )
                        
                        processed_count += 1
                        logger.info(f"Successfully processed {doc['filename']}")
                    
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    logger.error(f"Error details: {str(e)}")
                    continue
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return 0

if __name__ == "__main__":
    print("Starting feature extraction...")
    
    try:
        extractor = TrafficFeatureExtractor()
        
        total_processed = 0
        batch_size = 10
        
        while True:
            processed = extractor.process_images(batch_size)
            if processed == 0:
                break
            total_processed += processed
            print(f"Processed {total_processed} images so far...")
        
        print(f"Feature extraction complete. Total processed: {total_processed}")
        
    except Exception as e:
        print(f"Execution error: {str(e)}")