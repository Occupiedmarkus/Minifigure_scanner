# scripts/2_label_images.py
import cv2
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import json
from tqdm import tqdm
import numpy as np

class ImageLabeler:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup paths from environment variables
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.images_dir = self.base_dir / 'images'
        self.labels_dir = self.base_dir / 'labels'
        self.metadata_dir = self.base_dir / 'metadata'
        
        # Create necessary directories
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state management
        self.state_file = self.base_dir / 'labeling_state.json'
        self.current_image = None
        self.current_bbox = None
        self.drawing = False
        
        self.setup_logging()
        self.load_state()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'labeling.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_state(self):
        """Load labeling progress state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'labeled_images': [],
                'skipped_images': [],
                'last_position': 0
            }

    def save_state(self):
        """Save current labeling progress"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def draw_instructions(self, img):
        """Draw instructions on the image"""
        instructions = [
            "Controls:",
            "Left Click & Drag: Draw box",
            "S: Save and continue",
            "R: Reset current box",
            "Q: Quit",
            "Space: Skip image"
        ]
        
        y = 30
        for instruction in instructions:
            cv2.putText(img, instruction, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        
        return img

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding box"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_bbox = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_image.copy()
                cv2.rectangle(img_copy, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                self.draw_instructions(img_copy)
                cv2.imshow('Image Labeler', img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if abs(x - self.start_x) > 20 and abs(y - self.start_y) > 20:  # Minimum box size
                self.current_bbox = (min(self.start_x, x), min(self.start_y, y),
                                   abs(x - self.start_x), abs(y - self.start_y))
                cv2.rectangle(self.current_image, 
                            (self.current_bbox[0], self.current_bbox[1]),
                            (self.current_bbox[0] + self.current_bbox[2], 
                             self.current_bbox[1] + self.current_bbox[3]),
                            (0, 255, 0), 2)
                self.draw_instructions(self.current_image)
                cv2.imshow('Image Labeler', self.current_image)

    def label_image(self, image_path):
        """Label a single image"""
        try:
            # Read image and metadata
            img = cv2.imread(str(image_path))
            if img is None:
                self.logger.error(f"Could not read image: {image_path}")
                return False

            # Get metadata if available
            metadata_file = self.metadata_dir / f"{image_path.stem}.yaml"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                title = f"Labeling: {metadata.get('name', image_path.name)}"
            else:
                title = f"Labeling: {image_path.name}"

            # Setup window
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            height, width = img.shape[:2]
            self.current_image = img.copy()
            self.draw_instructions(self.current_image)
            
            # Set callback
            cv2.setMouseCallback(title, self.mouse_callback)

            while True:
                cv2.imshow(title, self.current_image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):  # Save
                    if self.current_bbox:
                        # Convert to YOLO format
                        x_center = (self.current_bbox[0] + self.current_bbox[2]/2) / width
                        y_center = (self.current_bbox[1] + self.current_bbox[3]/2) / height
                        w = self.current_bbox[2] / width
                        h = self.current_bbox[3] / height

                        # Save label
                        label_path = self.labels_dir / f"{image_path.stem}.txt"
                        with open(label_path, 'w') as f:
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

                        self.state['labeled_images'].append(image_path.name)
                        self.save_state()
                        break
                    else:
                        print("Please draw a bounding box first")

                elif key == ord('r'):  # Reset
                    self.current_image = img.copy()
                    self.draw_instructions(self.current_image)
                    self.current_bbox = None

                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    return False

                elif key == ord(' '):  # Skip
                    self.state['skipped_images'].append(image_path.name)
                    self.save_state()
                    break

            cv2.destroyAllWindows()
            return True

        except Exception as e:
            self.logger.error(f"Error labeling {image_path}: {e}")
            return False

    def label_all_images(self):
        """Label all images in the dataset"""
        # Get all images
        image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Filter out already labeled and skipped images
        unlabeled_images = [
            img for img in image_files 
            if img.name not in self.state['labeled_images'] 
            and img.name not in self.state['skipped_images']
        ]

        if not unlabeled_images:
            self.logger.info("No unlabeled images found")
            return

        self.logger.info(f"Found {len(unlabeled_images)} unlabeled images")
        
        # Start from last position
        start_idx = self.state['last_position']
        for idx, img_path in enumerate(unlabeled_images[start_idx:], start_idx):
            print(f"\nProcessing image {idx + 1}/{len(unlabeled_images)}")
            print(f"File: {img_path.name}")
            
            if not self.label_image(img_path):
                self.state['last_position'] = idx
                self.save_state()
                break
            
            self.state['last_position'] = idx
            self.save_state()

        # Print summary
        print("\nLabeling session summary:")
        print(f"Total images labeled: {len(self.state['labeled_images'])}")
        print(f"Total images skipped: {len(self.state['skipped_images'])}")

def main():
    labeler = ImageLabeler()
    labeler.label_all_images()

if __name__ == "__main__":
    main()