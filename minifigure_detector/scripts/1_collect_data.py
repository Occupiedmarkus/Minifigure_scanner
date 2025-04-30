import os
import time
import json
import logging
import requests
import yaml
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

class MinifigureDataCollector:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup paths and API
        self.api_key = os.getenv('REBRICKABLE_API_KEY')
        self.base_url = 'https://rebrickable.com/api/v3/lego'
        self.base_dir = Path(os.getenv('DATASET_PATH', 'dataset'))
        self.metadata_dir = self.base_dir / 'metadata'
        self.images_dir = self.base_dir / 'images' / 'train'
        
        # Create necessary directories
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            raise ValueError("API key not found. Please set REBRICKABLE_API_KEY in .env file")

    def get_existing_minifigs(self):
        """Get list of existing minifigure IDs from metadata directory"""
        if not self.metadata_dir.exists():
            return set()
        return {p.stem for p in self.metadata_dir.glob('*.yaml')}

    def extract_base_fig_id(self, filename):
        """Extract the base figure ID from a filename"""
        # Split by underscore and take the first part
        base_id = filename.split('_')[0]
        # Remove file extension if present
        base_id = base_id.split('.')[0]
        return base_id

    def get_minifig_data(self, fig_id):
        """Get minifigure data from API"""
        try:
            response = self.make_api_request(f"{self.base_url}/minifigs/{fig_id}/")
            if response:
                return response
        except Exception as e:
            self.logger.warning(f"Could not get data for {fig_id}: {e}")
        return None

    def make_api_request(self, url, headers=None, params=None, retry_count=3, base_delay=1):
        """Handle rate limits with exponential backoff"""
        if headers is None:
            headers = {'Authorization': f'key {self.api_key}'}
        
        for attempt in range(retry_count):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 429:  # Rate limit hit
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                if attempt == retry_count - 1:
                    raise
                wait_time = base_delay * (2 ** attempt)
                self.logger.warning(f"Request failed, retrying in {wait_time} seconds... Error: {e}")
                time.sleep(wait_time)
        
        return None

    def recreate_metadata(self, image_path, metadata_path):
        """Recreate metadata with support for multiple images per figure"""
        # Get the base figure ID
        base_fig_id = self.extract_base_fig_id(os.path.basename(image_path))
        
        try:
            metadata = self.get_minifig_data(base_fig_id)
            if metadata:
                # Save metadata using the full image filename
                save_path = os.path.join(metadata_path, f"{os.path.basename(image_path)}.yaml")
                
                formatted_metadata = {
                    'minifigure': {
                        'set_num': metadata.get('set_num', base_fig_id),
                        'name': metadata.get('name', 'Unknown'),
                        'year': metadata.get('year', -1),
                        'theme': metadata.get('theme_id', 'unknown'),
                        'category': 'official'
                    }
                }
                
                with open(save_path, 'w') as f:
                    yaml.dump(formatted_metadata, f)
                return True
        except Exception as e:
            self.logger.warning(f"Could not get data for {base_fig_id} from API: {e}")
        return False
      # Add path for custom minifigures file
        self.custom_figs_file = self.base_dir / 'labels' / 'custom_minifigures.yaml'
        
    def load_custom_minifigures(self):
        """Load custom minifigures configuration"""
        if self.custom_figs_file.exists():
            try:
                with open(self.custom_figs_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.warning(f"Could not load custom minifigures: {e}")
        return {}

    def check_orphaned_metadata(self, image_folder, metadata_folder):
        """Check for orphaned metadata files with support for multiple images"""
        metadata_files = set(f.replace('.yaml', '') for f in os.listdir(metadata_folder) 
                           if f.endswith('.yaml'))
        image_files = set()
        
        # Get all image files, including those in subdirectories
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.add(os.path.splitext(file)[0])

        # Filter out metadata files that belong to custom figures
        orphaned = set()
        for metadata_file in metadata_files:
            # Skip if filename starts with 'custom-'
            if metadata_file.startswith('custom-'):
                continue
                
            # Check if there's a corresponding image
            if metadata_file not in image_files:
                orphaned.add(metadata_file)

        if orphaned:
            self.logger.info("\nFound metadata files without corresponding images:")
            for file in sorted(orphaned):
                self.logger.info(f"- {file}")

        return orphaned

    def clean_orphaned_data(self, image_folder, metadata_folder):
        """Clean orphaned data with support for multiple images and custom figures"""
        orphaned = self.check_orphaned_metadata(image_folder, metadata_folder)
        cleaned = 0

        for file in orphaned:
            # Skip if filename starts with 'custom-'
            if file.startswith('custom-'):
                continue
                
            # Remove orphaned metadata
            metadata_path = os.path.join(metadata_folder, f"{file}.yaml")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                cleaned += 1

        return cleaned


    def get_official_minifigs(self, min_year=1975, limit=None):
        """Get official LEGO minifigures with maximum batch size"""
        MAX_PAGE_SIZE = 1000  # Maximum items per page
        MAX_BATCH_SIZE = 50   # Maximum minifigs per batch request
        
        headers = {'Authorization': f'key {self.api_key}'}
        minifigs = []
        page = 1
        
        existing_minifigs = self.get_existing_minifigs()
        self.logger.info(f"Found {len(existing_minifigs)} existing minifigures")
        
        pbar = tqdm(desc="Collecting minifigures", unit="minifigs")
        while True:
            try:
                if limit and len(minifigs) >= limit:
                    break
                
                # 1. Get maximum page size of minifigures
                params = {
                    'page_size': MAX_PAGE_SIZE,
                    'page': page,
                    'min_year': min_year,
                    'ordering': 'year'
                }
                
                data = self.make_api_request(f"{self.base_url}/minifigs/", params=params)
                if not data or not data.get('results'):
                    break
                
                # 2. Filter new minifigures
                new_figs = [
                    fig for fig in data['results']
                    if fig['set_num'] not in existing_minifigs
                    and (not limit or len(minifigs) < limit)
                ]
                
                # 3. Process in batches of 50
                for i in range(0, len(new_figs), MAX_BATCH_SIZE):
                    batch = new_figs[i:i + MAX_BATCH_SIZE]
                    set_nums = ','.join(fig['set_num'] for fig in batch)
                    
                    # Get batch details
                    batch_details = self.make_api_request(
                        f"{self.base_url}/minifigs/",
                        params={
                            'set_nums': set_nums,
                            'inc_parts': 0
                        }
                    )
                    
                    if batch_details and batch_details.get('results'):
                        for fig, details in zip(batch, batch_details['results']):
                            try:
                                # Process year and other details
                                year = -1
                                if details.get('year'):
                                    try:
                                        year = int(details['year'])
                                    except (ValueError, TypeError):
                                        pass
                                
                                fig.update({
                                    'year': year,
                                    'theme_id': details.get('theme_id', 'unknown'),
                                    'category_id': details.get('category_id', 'unknown')
                                })
                                
                                minifigs.append(fig)
                                pbar.update(1)
                                
                            except Exception as e:
                                self.logger.warning(f"Failed to process {fig['set_num']}: {e}")
                                continue
                    
                    # Add delay between batches to respect rate limits
                    time.sleep(1)
                
                if not data.get('next'):
                    break
                page += 1
                
            except Exception as e:
                self.logger.error(f"Error during batch processing: {e}")
                break
        
        pbar.close()
        return minifigs
    def download_image(self, url, path):
        """Download image from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            path.write_bytes(response.content)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to download image {url}: {e}")
            return False

    def save_metadata(self, minifig):
        """Save minifigure metadata"""
        try:
            metadata = {
                'minifigure': {
                    'set_num': minifig['set_num'],
                    'name': minifig['name'],
                    'year': minifig['year'],
                    'theme': minifig.get('theme_id', 'unknown'),
                    'category': 'official'
                }
            }
            
            metadata_file = self.metadata_dir / f"{minifig['set_num']}.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f)
            
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save metadata for {minifig['set_num']}: {e}")
            return False

    def collect_data(self, limit=None):
        """Main data collection process"""
        try:
            # Get minifigures
            minifigs = self.get_official_minifigs(limit=limit)
            
            # Download images and save metadata
            successful_downloads = 0
            with tqdm(minifigs, desc="Downloading images") as pbar:
                for minifig in pbar:
                    if minifig.get('set_img_url'):
                        image_path = self.images_dir / f"{minifig['set_num']}.png"
                        if not image_path.exists():
                            if self.download_image(minifig['set_img_url'], image_path):
                                successful_downloads += 1
                    
                    # Save metadata
                    self.save_metadata(minifig)
                    pbar.update(1)
            
            # Print summary
            total_minifigs = len(self.get_existing_minifigs())
            years = [m['year'] for m in minifigs if isinstance(m['year'], int) and m['year'] != -1]
            year_range = f"{min(years)}-{max(years)}" if years else "unknown"
            
            self.logger.info(f"""
Dataset collection completed:
- Total minifigures: {total_minifigs}
- New minifigures added: {len(minifigs)}
- Total images downloaded: {successful_downloads}
- Year range: {year_range}
""")
            
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            raise
def display_menu():
    """Display the main menu"""
    print("\nMinifigure Data Collection Tool")
    print("-------------------------------")
    print("1. Collect new minifigure data")
    print("2. Check dataset integrity")
    print("3. Clean orphaned metadata")
    print("4. View dataset statistics")
    print("5. Exit")
    return input("\nSelect an option (1-5): ")

def count_minifigures(collector):
    """Count official and custom minifigures"""
    official_count = 0
    custom_count = 0
    
    # Count from metadata files
    for metadata_file in collector.metadata_dir.glob('*.yaml'):
        if metadata_file.stem.startswith('custom-'):
            custom_count += 1
        else:
            official_count += 1
            
    return official_count, custom_count

def main():
    try:
        # Initialize collector
        collector = MinifigureDataCollector()
        
        while True:
            choice = display_menu()
            
            if choice == '1':
                print("\nCollecting new minifigure data...")
                collector.collect_data()
                
            elif choice == '2':
                print("\nChecking dataset integrity...")
                orphaned = collector.check_orphaned_metadata(
                    str(collector.images_dir),
                    str(collector.metadata_dir)
                )
                if not orphaned:
                    print("Dataset integrity verified - no orphaned files found")
                
            elif choice == '3':
                print("\nCleaning orphaned metadata files...")
                confirm = input("Are you sure you want to clean orphaned metadata? (y/n): ")
                if confirm.lower() == 'y':
                    cleaned = collector.clean_orphaned_data(
                        str(collector.images_dir),
                        str(collector.metadata_dir)
                    )
                    print(f"Cleaned {cleaned} orphaned metadata files")
                else:
                    print("Operation cancelled")
                
            elif choice == '4':
                # Get detailed statistics
                official_count, custom_count = count_minifigures(collector)
                total_images = len(list(collector.images_dir.glob('*.[pj][np][g]*')))
                
                print("\nDataset Statistics")
                print("-----------------")
                print(f"Official minifigures: {official_count}")
                print(f"Custom minifigures: {custom_count}")
                print(f"Total minifigures: {official_count + custom_count}")
                print(f"Total images: {total_images}")
                
            elif choice == '5':
                print("\nExiting...")
                break
                
            else:
                print("\nInvalid option. Please try again.")
            
            input("\nPress Enter to continue...")
            
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()