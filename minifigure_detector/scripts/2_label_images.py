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
        self.logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Debug: Print all found metadata files
        for f in metadata_files:
            self.logger.debug(f"Found metadata file: {f}")
        
        # Check both train directory and custom subdirectory
        standard_images = list(self.images_dir.glob('*.png')) + list(self.images_dir.glob('*.jpg'))
        self.logger.info(f"Found {len(standard_images)} standard images")
        
        custom_dir = self.images_dir / 'custom'
        if custom_dir.exists():
            custom_dirs = list(custom_dir.glob('custom-*'))
            self.logger.info(f"Found {len(custom_dirs)} custom directories:")
            for d in custom_dirs:
                custom_images = list(d.glob(f'{d.name}-*.png')) + list(d.glob(f'{d.name}-*.jpg'))
                self.logger.info(f"  - {d.name}: {len(custom_images)} images")
        
        metadata_list = []
        valid_count = 0
        invalid_count = 0
        
        for meta_file in tqdm(metadata_files, desc="Loading metadata"):
            try:
                with open(meta_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self.logger.debug(f"Loaded metadata content from {meta_file}:")
                    self.logger.debug(str(data))
                
                # Extract minifigure data
                metadata = data.get('minifigure', data)
                
                # Basic validation
                required_fields = ['set_num', 'name', 'year']
                missing_fields = [field for field in required_fields if field not in metadata]
                
                if not missing_fields:
                    # Handle year conversion
                    if metadata['year'] == 'unknown':
                        metadata['year'] = -1
                    elif isinstance(metadata['year'], str):
                        try:
                            metadata['year'] = int(metadata['year'])
                        except ValueError:
                            metadata['year'] = -1
                    
                    # Handle different image types
                    if metadata.get('is_custom') or 'custom-' in metadata.get('set_num', ''):
                        # For custom images
                        custom_dir = self.images_dir / 'custom' / metadata['set_num']
                        custom_images = list(custom_dir.glob(f"{metadata['set_num']}-*.png")) + \
                                    list(custom_dir.glob(f"{metadata['set_num']}-*.jpg"))
                        
                        if custom_images:
                            metadata['images'] = [str(img) for img in sorted(custom_images)]
                            self.logger.debug(f"Found {len(custom_images)} custom images for {metadata['set_num']}")
                            metadata_list.append(metadata)
                            valid_count += 1
                        else:
                            self.logger.warning(f"No images found for custom minifig {metadata['set_num']}")
                            invalid_count += 1
                    else:
                        # For standard images
                        image_path = self.images_dir / f"{metadata['set_num']}.png"
                        if not image_path.exists():
                            image_path = self.images_dir / f"{metadata['set_num']}.jpg"
                        
                        if image_path.exists():
                            metadata['images'] = [str(image_path)]
                            self.logger.debug(f"Found standard image: {image_path}")
                            metadata_list.append(metadata)
                            valid_count += 1
                        else:
                            self.logger.warning(f"Missing image for {metadata['set_num']}")
                            invalid_count += 1
                else:
                    self.logger.warning(f"Missing required fields {missing_fields} in {meta_file.name}")
                    invalid_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {meta_file.name}: {e}")
                invalid_count += 1
        
        # Final validation
        if not metadata_list:
            self.logger.error("No valid metadata entries found!")
            self.logger.error("Directory structure:")
            self.print_directory_structure()
        else:
            self.logger.info(f"""
    Metadata loading completed:
    - Valid entries: {valid_count}
    - Invalid entries: {invalid_count}
    - Custom minifigs: {sum(1 for m in metadata_list if m.get('is_custom', False))}
    - Official minifigs: {sum(1 for m in metadata_list if not m.get('is_custom', False))}
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
        
        # Convert numpy values to native Python types
        year_mapping = {}
        for year in self.year_encoder.classes_:
            year_val = int(year) if isinstance(year, (np.integer, int)) else str(year)
            encoded_val = int(self.year_encoder.transform([year])[0])
            year_mapping[year_val] = encoded_val

        theme_mapping = {}
        for theme in self.theme_encoder.classes_:
            theme_val = str(theme)
            encoded_val = int(self.theme_encoder.transform([theme])[0])
            theme_mapping[theme_val] = encoded_val

        # Create plain dictionary for saving
        encoders_mapping = {
            'year_mapping': year_mapping,
            'theme_mapping': theme_mapping
        }
        
        # Save in plain YAML format
        with open(self.labels_dir / 'encoders_mapping.yaml', 'w') as f:
            yaml.dump(encoders_mapping, f, default_flow_style=False)
        
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