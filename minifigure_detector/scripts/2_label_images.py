import os
import yaml
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv


class MinifigureDataPreprocessor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup paths
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.metadata_dir = self.base_dir / 'metadata'
        self.images_dir = self.base_dir / 'images'
        self.labels_dir = self.base_dir / 'labels'
        
        # Ensure directories exist
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize label encoders
        self.year_encoder = LabelEncoder()
        self.theme_encoder = LabelEncoder()
        
    def load_metadata(self):
        """Load and validate all metadata files"""
        metadata_files = list(self.metadata_dir.glob('*.yaml'))
        self.logger.info(f"Found {len(metadata_files)} metadata files")
        
        metadata_list = []
        valid_count = 0
        invalid_count = 0
        
        for meta_file in tqdm(metadata_files, desc="Loading metadata"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                # Basic validation
                required_fields = ['set_num', 'name', 'year']
                if all(field in metadata for field in required_fields):
                    # Convert 'unknown' year to a specific value for processing
                    if metadata['year'] == 'unknown':
                        metadata['year'] = -1  # Use -1 to represent unknown year
                    elif isinstance(metadata['year'], str):
                        try:
                            metadata['year'] = int(metadata['year'])
                        except ValueError:
                            metadata['year'] = -1
                    
                    # Ensure image exists
                    image_path = self.images_dir / f"{metadata['set_num']}.jpg"
                    if image_path.exists():
                        metadata_list.append(metadata)
                        valid_count += 1
                    else:
                        self.logger.warning(f"Missing image for {metadata['set_num']}")
                        invalid_count += 1
                else:
                    self.logger.warning(f"Missing required fields in {meta_file.name}")
                    invalid_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {meta_file.name}: {e}")
                invalid_count += 1
                
        self.logger.info(f"Successfully loaded {valid_count} valid metadata files")
        if invalid_count > 0:
            self.logger.warning(f"Found {invalid_count} invalid metadata files")
            
        return metadata_list

    def process_images(self, metadata_list, target_size=(224, 224)):
        """Process and normalize images"""
        image_data = []
        valid_metadata = []
        
        for metadata in tqdm(metadata_list, desc="Processing images"):
            try:
                image_path = self.images_dir / f"{metadata['set_num']}.jpg"
                
                # Load and preprocess image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img) / 255.0
                    
                    image_data.append(img_array)
                    valid_metadata.append(metadata)
                    
            except Exception as e:
                self.logger.error(f"Error processing image {metadata['set_num']}: {e}")
                
        return np.array(image_data), valid_metadata

    def encode_labels(self, metadata_list):
        """Encode categorical labels"""
        # Extract years and themes
        years = [m['year'] for m in metadata_list]
        themes = [m.get('theme', 'unknown') for m in metadata_list]
        
        # Fit encoders
        self.year_encoder.fit(years)
        self.theme_encoder.fit(themes)
        
        # Encode labels
        encoded_years = self.year_encoder.transform(years)
        encoded_themes = self.theme_encoder.transform(themes)
        
        # Save encoders mapping
        encoders_mapping = {
            'year_mapping': dict(zip(self.year_encoder.classes_, self.year_encoder.transform(self.year_encoder.classes_))),
            'theme_mapping': dict(zip(self.theme_encoder.classes_, self.theme_encoder.transform(self.theme_encoder.classes_)))
        }
        
        with open(self.labels_dir / 'encoders_mapping.yaml', 'w') as f:
            yaml.dump(encoders_mapping, f)
        
        return encoded_years, encoded_themes

    def preprocess_data(self, target_size=(224, 224)):
        """Main preprocessing pipeline"""
        try:
            # Load and validate metadata
            metadata_list = self.load_metadata()
            if not metadata_list:
                raise ValueError("No valid metadata found")
            
            # Process images
            images, valid_metadata = self.process_images(metadata_list, target_size)
            if len(images) == 0:
                raise ValueError("No valid images processed")
            
            # Encode labels
            encoded_years, encoded_themes = self.encode_labels(valid_metadata)
            
            # Save preprocessed data
            np.save(self.labels_dir / 'images.npy', images)
            np.save(self.labels_dir / 'years.npy', encoded_years)
            np.save(self.labels_dir / 'themes.npy', encoded_themes)
            
            # Save metadata mapping
            metadata_mapping = {m['set_num']: {
                'name': m['name'],
                'year': m['year'],
                'theme': m.get('theme', 'unknown')
            } for m in valid_metadata}
            
            with open(self.labels_dir / 'metadata_mapping.yaml', 'w') as f:
                yaml.dump(metadata_mapping, f)
            
            self.logger.info(f"""
Preprocessing completed successfully:
- Processed images: {len(images)}
- Unique years: {len(set(encoded_years))}
- Unique themes: {len(set(encoded_themes))}
- Image shape: {images[0].shape}
""")
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

def main():
    preprocessor = MinifigureDataPreprocessor()
    preprocessor.preprocess_data()

if __name__ == "__main__":
    main()