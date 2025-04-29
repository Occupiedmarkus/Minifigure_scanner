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
        self.images_dir = self.base_dir / 'images'/'train'
        self.labels_dir = self.base_dir / 'labels'
        
        # Ensure directories exist
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize label encoders
        self.year_encoder = LabelEncoder()
        self.theme_encoder = LabelEncoder()
        
    def load_metadata(self):
        """Load and validate all metadata files"""
        metadata_files = list(self.metadata_dir.glob('*.yaml'))
        
        # Check both train directory and custom subdirectory
        standard_images = list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg'))
        custom_dir = self.images_dir / 'custom'
        
        if custom_dir.exists():
            # Look for custom-(name) directories
            custom_dirs = list(custom_dir.glob('custom-*'))
            self.logger.info(f"Found {len(custom_dirs)} custom minifigure directories")
            
            # Create metadata for custom images if not exists
            for custom_dir in custom_dirs:
                minifig_id = custom_dir.name  # e.g., 'custom-batman'
                metadata_file = self.metadata_dir / f"{minifig_id}.yaml"
                
                if not metadata_file.exists():
                    # Get all images for this custom minifigure
                    custom_images = list(custom_dir.glob(f'{minifig_id}-*.png')) + \
                                list(custom_dir.glob(f'{minifig_id}-*.jpg'))
                    
                    if custom_images:
                        metadata = {
                            'minifigure': {
                                'set_num': minifig_id,
                                'name': minifig_id.replace('custom-', ''),
                                'year': -1,  # Unknown year
                                'theme': 'custom',
                                'category': 'custom',
                                'is_custom': True,
                                'image_paths': [str(img) for img in sorted(custom_images)]
                            }
                        }
                        with open(metadata_file, 'w') as f:
                            yaml.dump(metadata, f)
                        metadata_files.append(metadata_file)
        
        self.logger.info(f"Found {len(metadata_files)} total metadata files")
        
        metadata_list = []
        valid_count = 0
        invalid_count = 0
        
        for meta_file in tqdm(metadata_files, desc="Loading metadata"):
            try:
                with open(meta_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                metadata = data.get('minifigure', data)
                
                # Basic validation
                required_fields = ['set_num', 'name', 'year']
                if all(field in metadata for field in required_fields):
                    # Handle year conversion
                    if metadata['year'] == 'unknown':
                        metadata['year'] = -1
                    elif isinstance(metadata['year'], str):
                        try:
                            metadata['year'] = int(metadata['year'])
                        except ValueError:
                            metadata['year'] = -1
                    
                    # Handle different image types
                    if metadata.get('is_custom'):
                        # For custom images, use the stored image paths
                        if 'image_paths' in metadata:
                            metadata['images'] = [
                                str(path) for path in metadata['image_paths']
                                if Path(path).exists()
                            ]
                            if metadata['images']:
                                metadata_list.append(metadata)
                                valid_count += 1
                                self.logger.debug(f"Valid custom metadata found for {metadata['set_num']}")
                            else:
                                self.logger.warning(f"No valid images found for {metadata['set_num']}")
                                invalid_count += 1
                    else:
                        # For standard images in train directory
                        image_path = self.images_dir / f"{metadata['set_num']}.png"
                        if not image_path.exists():
                            image_path = self.images_dir / f"{metadata['set_num']}.jpg"
                        
                        if image_path.exists():
                            metadata['images'] = [str(image_path)]
                            metadata_list.append(metadata)
                            valid_count += 1
                        else:
                            self.logger.warning(f"Missing image for {metadata['set_num']}")
                            invalid_count += 1
                else:
                    missing_fields = [field for field in required_fields if field not in metadata]
                    self.logger.warning(f"Missing required fields {missing_fields} in {meta_file.name}")
                    invalid_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {meta_file.name}: {e}")
                invalid_count += 1
        
        self.logger.info(f"""
    Metadata loading completed:
    - Valid entries: {valid_count}
    - Invalid entries: {invalid_count}
    - Custom images: {sum(1 for m in metadata_list if m.get('is_custom', True))}
    - Standard images: {sum(1 for m in metadata_list if not m.get('is_custom', False))}
    """)
        
        return metadata_list

    def process_images(self, metadata_list, target_size=(224, 224)):
        """Process and normalize images"""
        image_data = []
        valid_metadata = []
        failed_images = []
        
        for metadata in tqdm(metadata_list, desc="Processing images"):
            try:
                # Process each image for this minifigure
                for image_path in metadata['images']:
                    image_path = Path(image_path)
                    
                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img).astype(np.float32) / 255.0
                        
                        image_data.append(img_array)
                        valid_metadata.append(metadata)
                        
                    self.logger.debug(f"Successfully processed {image_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing image for {metadata['set_num']}: {e}")
                failed_images.append(metadata['set_num'])
        
        if not image_data:
            raise ValueError("No images were successfully processed")
        
        processed_images = np.array(image_data)
        
        self.logger.info(f"""
    Image processing completed:
    - Successfully processed: {len(processed_images)} images
    - Failed to process: {len(failed_images)} images
    - Image shape: {processed_images[0].shape}
    - Memory usage: {processed_images.nbytes / 1024 / 1024:.2f} MB
    """)

        if failed_images:
            self.logger.warning("Failed images: " + ", ".join(failed_images))
        
        return processed_images, valid_metadata
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