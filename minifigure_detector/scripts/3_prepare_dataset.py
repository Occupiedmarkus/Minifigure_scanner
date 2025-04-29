# scripts/3_prepare_dataset.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import shutil
import yaml
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import numpy as np

class DatasetPreparator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup paths
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.splits = {
            'train': self.base_dir / 'train',
            'val': self.base_dir / 'val',
            'test': self.base_dir / 'test'
        }
        
        # Source directories
        self.images_dir = self.base_dir / 'images'
        self.labels_dir = self.base_dir / 'labels'
        self.metadata_dir = self.base_dir / 'metadata'
        
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'dataset_preparation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories for dataset splits"""
        for split in self.splits.values():
            (split / 'images').mkdir(parents=True, exist_ok=True)
            (split / 'labels').mkdir(parents=True, exist_ok=True)

    def validate_data(self):
        """Validate that each image has a corresponding label file"""
        images = list(self.images_dir.glob('*.jpg'))
        valid_pairs = []
        
        self.logger.info("Validating image-label pairs...")
        for img_path in tqdm(images, desc="Validating data"):
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                # Validate label format
                try:
                    with open(label_path, 'r') as f:
                        label_content = f.read().strip()
                        # Check YOLO format: class x_center y_center width height
                        values = label_content.split()
                        if len(values) == 5 and all(self.is_float(v) for v in values[1:]):
                            valid_pairs.append((img_path, label_path))
                        else:
                            self.logger.warning(f"Invalid label format in {label_path}")
                except Exception as e:
                    self.logger.error(f"Error reading {label_path}: {e}")
            else:
                self.logger.warning(f"No label found for {img_path}")
        
        return valid_pairs

    @staticmethod
    def is_float(value):
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def augment_image(self, image_path, label_path):
        """Apply basic augmentations to image and adjust labels"""
        augmented_pairs = []
        img = cv2.imread(str(image_path))
        
        if img is None:
            return augmented_pairs

        # Read original label
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
            class_id, x_center, y_center, width, height = map(float, label_content.split())

        # Basic augmentations
        augmentations = [
            ('flip', cv2.flip(img, 1)),  # Horizontal flip
            ('bright', cv2.convertScaleAbs(img, alpha=1.2, beta=10)),  # Brightness
            ('contrast', cv2.convertScaleAbs(img, alpha=1.3, beta=0))  # Contrast
        ]

        for aug_name, aug_img in augmentations:
            # Create augmented image filename
            aug_img_path = image_path.parent / f"{image_path.stem}_{aug_name}.jpg"
            aug_label_path = label_path.parent / f"{image_path.stem}_{aug_name}.txt"

            # Save augmented image
            cv2.imwrite(str(aug_img_path), aug_img)

            # Adjust and save label (only horizontal flip needs adjustment)
            with open(aug_label_path, 'w') as f:
                if aug_name == 'flip':
                    # Adjust x coordinate for horizontal flip
                    new_x = 1 - x_center
                    f.write(f"{int(class_id)} {new_x} {y_center} {width} {height}\n")
                else:
                    f.write(label_content + '\n')

            augmented_pairs.append((aug_img_path, aug_label_path))

        return augmented_pairs

    def split_dataset(self, valid_pairs, train_size=0.7, val_size=0.2):
        """Split dataset into train/val/test sets"""
        # Optionally augment training data
        augment = input("Would you like to augment the training data? (y/n): ").lower() == 'y'
        
        # Split datasets
        train_val_pairs, test_pairs = train_test_split(
            valid_pairs, 
            train_size=train_size + val_size,
            random_state=42
        )
        
        train_pairs, val_pairs = train_test_split(
            train_val_pairs,
            train_size=train_size/(train_size + val_size),
            random_state=42
        )

        # Augment training data if requested
        if augment:
            self.logger.info("Augmenting training data...")
            augmented_pairs = []
            for img_path, label_path in tqdm(train_pairs, desc="Augmenting"):
                aug_pairs = self.augment_image(img_path, label_path)
                augmented_pairs.extend(aug_pairs)
            train_pairs.extend(augmented_pairs)
            self.logger.info(f"Added {len(augmented_pairs)} augmented images")

        # Copy files to respective directories
        splits_data = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }

        for split_name, pairs in splits_data.items():
            self.logger.info(f"Preparing {split_name} split...")
            split_dir = self.splits[split_name]
            
            for img_path, label_path in tqdm(pairs, desc=f"Copying {split_name}"):
                # Copy image
                shutil.copy2(img_path, split_dir / 'images' / img_path.name)
                # Copy label
                shutil.copy2(label_path, split_dir / 'labels' / label_path.name)

        # Create data.yaml
        self.create_data_yaml(len(splits_data['train']), 
                            len(splits_data['val']), 
                            len(splits_data['test']))

    def create_data_yaml(self, train_count, val_count, test_count):
        """Create data.yaml file for YOLOv8"""
        yaml_content = {
            'path': str(self.base_dir.absolute()),
            'train': str(Path('train/images')),
            'val': str(Path('val/images')),
            'test': str(Path('test/images')),
            'nc': 1,
            'names': ['minifigure'],
            'splits': {
                'train': train_count,
                'val': val_count,
                'test': test_count
            }
        }
        
        yaml_path = self.base_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.logger.info(f"Created data.yaml at {yaml_path}")

    def prepare_dataset(self):
        """Main method to prepare the dataset"""
        try:
            # Validate data
            self.logger.info("Starting dataset preparation...")
            valid_pairs = self.validate_data()
            
            if not valid_pairs:
                self.logger.error("No valid image-label pairs found!")
                return
            
            self.logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
            
            # Get split ratios from user or use defaults
            train_size = float(input("Enter train split ratio (default 0.7): ") or 0.7)
            val_size = float(input("Enter validation split ratio (default 0.2): ") or 0.2)
            test_size = 1 - train_size - val_size
            
            self.logger.info(f"Split ratios - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            
            # Split and prepare dataset
            self.split_dataset(valid_pairs, train_size, val_size)
            
            self.logger.info("Dataset preparation completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise

def main():
    preparator = DatasetPreparator()
    preparator.prepare_dataset()

if __name__ == "__main__":
    main()