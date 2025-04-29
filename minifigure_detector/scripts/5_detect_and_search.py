# scripts/5_detect_and_search.py
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import yaml
import json
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
from tqdm import tqdm

class MinifigureDetector:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup paths
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.model_path = Path(os.getenv('MODEL_PATH', 'models/weights/best.pt'))
        self.db_path = Path(os.getenv('DATABASE_PATH', 'database/minifigures.db'))
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup components
        self.setup_logging()
        self.setup_database()
        self.load_model()
        self.load_metadata()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'detection.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup SQLite database for detection results"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                detection_time TIMESTAMP,
                confidence REAL,
                bbox_x REAL,
                bbox_y REAL,
                bbox_width REAL,
                bbox_height REAL,
                metadata_id TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                name TEXT,
                year INTEGER,
                num_parts INTEGER,
                url TEXT
            )
        ''')
        
        self.conn.commit()

    def load_model(self):
        """Load the trained YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def load_metadata(self):
        """Load minifigure metadata"""
        self.metadata = {}
        metadata_dir = self.base_dir / 'metadata'
        
        if metadata_dir.exists():
            for meta_file in metadata_dir.glob('*.yaml'):
                with open(meta_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self.metadata[meta_file.stem] = data
                    
                    # Update database
                    self.cursor.execute('''
                        INSERT OR REPLACE INTO metadata (id, name, year, num_parts, url)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        data['set_num'],
                        data['name'],
                        data['year'],
                        data['num_parts'],
                        data['url']
                    ))
            
            self.conn.commit()
            self.logger.info(f"Loaded metadata for {len(self.metadata)} minifigures")

    def detect_image(self, image_path, conf_threshold=0.25):
        """Detect minifigures in an image"""
        try:
            # Run detection
            results = self.model(image_path, conf=conf_threshold)[0]
            
            # Process results
            detections = []
            for box in results.boxes:
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
                
                # Convert to center format
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                detection = {
                    'confidence': confidence,
                    'bbox': (x_center, y_center, width, height)
                }
                detections.append(detection)
                
                # Store in database
                self.cursor.execute('''
                    INSERT INTO detections 
                    (image_path, detection_time, confidence, bbox_x, bbox_y, bbox_width, bbox_height)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(image_path),
                    datetime.now(),
                    confidence,
                    x_center,
                    y_center,
                    width,
                    height
                ))
            
            self.conn.commit()
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return []

    def draw_detections(self, image_path, detections, save_path=None):
        """Draw detection boxes on image"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            height, width = img.shape[:2]
            
            # Draw each detection
            for det in detections:
                # Convert center format to corners
                x_center, y_center, w, h = det['bbox']
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                conf_text = f"{det['confidence']:.2f}"
                cv2.putText(img, conf_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save or display
            if save_path:
                cv2.imwrite(str(save_path), img)
            else:
                cv2.imshow('Detections', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            self.logger.error(f"Error drawing detections: {e}")

    def search_similar(self, image_path, top_k=5):
        """Search for similar minifigures in the dataset"""
        try:
            # Get features from the query image
            results = self.model(image_path, conf=0.25)[0]
            if len(results.boxes) == 0:
                return []
            
            # Get the highest confidence detection
            query_features = results.boxes[0].xywh[0].tolist()  # center format
            
            # Search database
            self.cursor.execute('''
                SELECT d.image_path, d.confidence, m.name, m.year, m.url,
                       ABS(d.bbox_width - ?) + ABS(d.bbox_height - ?) as diff
                FROM detections d
                LEFT JOIN metadata m ON d.metadata_id = m.id
                WHERE d.confidence > 0.5
                ORDER BY diff ASC
                LIMIT ?
            ''', (query_features[2], query_features[3], top_k))
            
            similar = self.cursor.fetchall()
            
            # Format results
            results = []
            for row in similar:
                results.append({
                    'image_path': row[0],
                    'confidence': row[1],
                    'name': row[2],
                    'year': row[3],
                    'url': row[4],
                    'similarity_score': 1.0 / (1.0 + row[5])  # Convert difference to similarity
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during similarity search: {e}")
            return []

    def batch_process(self, input_dir, output_dir=None):
        """Process multiple images in a directory"""
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        results = {}
        for img_path in tqdm(image_files, desc="Processing images"):
            detections = self.detect_image(img_path)
            results[str(img_path)] = detections
            
            if output_dir:
                output_file = output_path / f"{img_path.stem}_detected{img_path.suffix}"
                self.draw_detections(img_path, detections, save_path=output_file)
        
        return results

    def interactive_mode(self):
        """Interactive detection and search mode"""
        while True:
            print("\nMinifigure Detection and Search")
            print("1. Detect in single image")
            print("2. Batch process directory")
            print("3. Search similar minifigures")
            print("4. View detection history")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                image_path = input("Enter image path: ")
                if not Path(image_path).exists():
                    print("Image not found!")
                    continue
                    
                detections = self.detect_image(image_path)
                self.draw_detections(image_path, detections)
                
            elif choice == '2':
                input_dir = input("Enter input directory: ")
                output_dir = input("Enter output directory (optional): ")
                
                if not Path(input_dir).exists():
                    print("Directory not found!")
                    continue
                    
                results = self.batch_process(input_dir, output_dir)
                print(f"Processed {len(results)} images")
                
            elif choice == '3':
                image_path = input("Enter query image path: ")
                if not Path(image_path).exists():
                    print("Image not found!")
                    continue
                    
                similar = self.search_similar(image_path)
                
                print("\nSimilar minifigures:")
                for idx, result in enumerate(similar, 1):
                    print(f"\n{idx}. {result['name']} ({result['year']})")
                    print(f"   Confidence: {result['confidence']:.2f}")
                    print(f"   Similarity: {result['similarity_score']:.2f}")
                    print(f"   URL: {result['url']}")
                
            elif choice == '4':
                self.cursor.execute('''
                    SELECT image_path, detection_time, confidence
                    FROM detections
                    ORDER BY detection_time DESC
                    LIMIT 10
                ''')
                
                print("\nRecent detections:")
                for row in self.cursor.fetchall():
                    print(f"\nImage: {row[0]}")
                    print(f"Time: {row[1]}")
                    print(f"Confidence: {row[2]:.2f}")
                
            elif choice == '5':
                break
                
            else:
                print("Invalid choice!")

    def cleanup(self):
        """Cleanup resources"""
        self.conn.close()

def main():
    detector = MinifigureDetector()
    try:
        detector.interactive_mode()
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()