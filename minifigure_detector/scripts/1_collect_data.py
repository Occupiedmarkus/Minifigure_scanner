# scripts/1_collect_data.py
import os
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key
from concurrent.futures import ThreadPoolExecutor
import yaml
from tqdm import tqdm
import time

class MinifigureDataCollector:
    def __init__(self):
        self.setup_env()
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.images_dir = self.base_dir / 'images'
        self.setup_logging()
        
    def setup_env(self):
        """Setup and validate environment variables"""
        load_dotenv()
        
        # Check if API key exists
        self.api_key = os.getenv('REBRICKABLE_API_KEY')
        if not self.api_key:
            self.api_key = input("Enter your Rebrickable API key: ").strip()
            # Save API key to .env file
            env_path = Path('.env')
            set_key(env_path, 'REBRICKABLE_API_KEY', self.api_key)
            print("API key saved to .env file")

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = Path('logs')
        log_file.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file / 'data_collection.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_api_key(self):
        """Validate the Rebrickable API key"""
        headers = {"Authorization": f"key {self.api_key}"}
        test_url = "https://rebrickable.com/api/v3/lego/minifigs/?page_size=1"
        
        try:
            response = requests.get(test_url, headers=headers)
            response.raise_for_status()
            self.logger.info("API key validated successfully")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

    def download_image(self, minifig):
        """Download a single minifigure image"""
        headers = {"Authorization": f"key {self.api_key}"}
        
        try:
            if not minifig.get('set_img_url'):
                return None

            time.sleep(0.5)  # Rate limiting
            img_response = requests.get(minifig['set_img_url'], 
                                      headers=headers, 
                                      timeout=10)
            img_response.raise_for_status()

            img_path = self.images_dir / f"{minifig['set_num']}.jpg"
            
            with open(img_path, 'wb') as f:
                f.write(img_response.content)

            # Save minifig metadata
            metadata_path = self.base_dir / 'metadata' / f"{minifig['set_num']}.yaml"
            metadata_path.parent.mkdir(exist_ok=True)
            with open(metadata_path, 'w') as f:
                yaml.dump({
                    'set_num': minifig['set_num'],
                    'name': minifig['name'],
                    'year': minifig['year'],
                    'num_parts': minifig['num_parts'],
                    'url': minifig['set_url']
                }, f)

            return minifig['set_num']

        except Exception as e:
            self.logger.error(f"Error downloading {minifig['set_num']}: {e}")
            return None

    def collect_minifig_data(self, num_samples=100):
        """Collect minifigure data from Rebrickable"""
        if not self.validate_api_key():
            self.logger.error("Invalid API key. Please check your .env file")
            return False
        
        try:
            # Create necessary directories
            self.images_dir.mkdir(parents=True, exist_ok=True)
            (self.base_dir / 'labels').mkdir(parents=True, exist_ok=True)

            # Get minifigure list
            headers = {"Authorization": f"key {self.api_key}"}
            params = {
                "page_size": num_samples,
                "ordering": "-num_parts",
                "min_parts": 3
            }
            
            response = requests.get(
                "https://rebrickable.com/api/v3/lego/minifigs/",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            minifigs = response.json()['results']

            self.logger.info(f"Found {len(minifigs)} minifigures")

            # Download images with progress bar
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.download_image, minifig) 
                          for minifig in minifigs]
                
                with tqdm(total=len(minifigs), desc="Downloading images") as pbar:
                    for future in futures:
                        result = future.result()
                        if result:
                            pbar.update(1)

            # Create data.yaml
            self.create_data_yaml()
            
            return True

        except Exception as e:
            self.logger.error(f"Collection failed: {e}")
            return False

    def create_data_yaml(self):
        """Create data.yaml file for YOLOv8"""
        yaml_content = {
            'path': str(self.base_dir.absolute()),
            'train': str(Path('train/images')),
            'val': str(Path('val/images')),
            'test': str(Path('test/images')),
            'nc': 1,
            'names': ['minifigure']
        }
        
        yaml_path = self.base_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.logger.info(f"Created data.yaml at {yaml_path}")

def main():
    collector = MinifigureDataCollector()
    
    # Number of minifigures to collect
    num_samples = int(input("Enter number of minifigures to collect (default 100): ") or 100)
    collector.collect_minifig_data(num_samples=num_samples)

if __name__ == "__main__":
    main()